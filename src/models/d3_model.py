import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from einops import rearrange
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
from omegaconf import OmegaConf

from . import rotary
from .fused_add_dropout_scale import (
    bias_dropout_add_scale_fused_train,
    bias_dropout_add_scale_fused_inference,
    modulate_fused,
)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


def residual_linear(x, W, x_skip, residual_scale):
    """x_skip + residual_scale * W @ x"""
    dim_out, dim_in = W.shape[0], W.shape[1]
    return torch.addmm(
        x_skip.view(-1, dim_out), x.view(-1, dim_in), W.T, alpha=residual_scale
    ).view(*x.shape[:-1], dim_out)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, silu=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core Model                                    #
#################################################################################


class DDiTBlock(nn.Module):

    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )
        self.dropout2 = nn.Dropout(dropout)

        self.dropout = dropout

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

        # For storing attention scores
        self.attention_scores = None
        self.save_attention = False

    def _get_bias_dropout_scale(self):
        return (
            bias_dropout_add_scale_fused_train
            if self.training
            else bias_dropout_add_scale_fused_inference
        )

    def forward(self, x, rotary_cos_sin, c, seqlens=None):
        batch_size, seq_len = x.shape[0], x.shape[1]

        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
        )

        # attention operation
        x_skip = x
        x = modulate_fused(self.norm1(x), shift_msa, scale_msa)

        qkv = self.attn_qkv(x)
        qkv = rearrange(
            qkv, "b s (three h d) -> b s three h d", three=3, h=self.n_heads
        )
        with torch.cuda.amp.autocast(enabled=False):
            cos, sin = rotary_cos_sin
            qkv = rotary.apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
        qkv = rearrange(qkv, "b s ... -> (b s) ...")
        if seqlens is None:
            cu_seqlens = torch.arange(
                0,
                (batch_size + 1) * seq_len,
                step=seq_len,
                dtype=torch.int32,
                device=qkv.device,
            )
        else:
            cu_seqlens = seqlens.cumsum(-1)

        # Compute attention scores if needed
        if self.save_attention:
            # Need to properly reshape qkv back to batch format first
            qkv_reshaped = rearrange(qkv, "(b s) ... -> b s ...", b=batch_size)

            # Extract q, k, v from reshaped tensor
            q_batch = qkv_reshaped[
                :, :, 0
            ]  # Shape: [batch_size, seq_len, n_heads, dim]
            k_batch = qkv_reshaped[
                :, :, 1
            ]  # Shape: [batch_size, seq_len, n_heads, dim]

            # Initialize tensor to store attention scores for all heads
            attn_scores = []

            # Calculate attention for each head
            for head_idx in range(self.n_heads):
                # Extract this head's queries and keys
                q_head = q_batch[:, :, head_idx]  # [batch_size, seq_len, dim]
                k_head = k_batch[:, :, head_idx]  # [batch_size, seq_len, dim]

                # Compute attention matrix: [batch_size, seq_len, seq_len]
                scale_factor = q_head.size(-1) ** 0.5
                attn = (
                    torch.bmm(
                        q_head.view(
                            batch_size, seq_len, -1
                        ),  # [batch_size, seq_len, dim]
                        k_head.view(batch_size, seq_len, -1).transpose(
                            1, 2
                        ),  # [batch_size, dim, seq_len]
                    )
                    / scale_factor
                )

                attn_scores.append(attn.unsqueeze(1))  # Add head dimension

            # Stack along head dimension
            self.attention_scores = torch.cat(
                attn_scores, dim=1
            )  # [batch_size, n_heads, seq_len, seq_len]

        # Flash attention for speed
        x = flash_attn_varlen_qkvpacked_func(
            qkv, cu_seqlens, seq_len, 0.0, causal=False
        )

        x = rearrange(x, "(b s) h d -> b s (h d)", b=batch_size)

        x = bias_dropout_scale_fn(
            self.attn_out(x), None, gate_msa, x_skip, self.dropout
        )

        # mlp operation
        x = bias_dropout_scale_fn(
            self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)),
            None,
            gate_mlp,
            x,
            self.dropout,
        )
        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        """
        Mode arg: 0 -> use a learned layer, 1 -> use eigenvectors,
        2-> add in eigenvectors, 3 -> use pretrained embedding matrix
        """
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        self.signal_embedding = nn.Linear(5, dim)  # Updated for 5 cell types
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x, y):
        # print (x.shape, x.dtype)
        # print (y.shape, y.dtype)
        vocab_embed = self.embedding[x]  # return only this if label embedding is used
        signal_embed = self.signal_embedding(y.to(torch.float32))
        # print (vocab_embed.shape)
        # print (signal_embed.shape)
        return torch.add(
            vocab_embed, signal_embed[:, None, :]
        )  # [:, None, :] extra for deepstarr


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate_fused(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class GaussianFourierProjection(nn.Module):
    """
    Gaussian random features for encoding time steps.
    """

    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """
    A fully connected layer that reshapes outputs to feature maps.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[...]


class SEDD(nn.Module):
    def __init__(self, config):
        super().__init__()

        # hack to make loading in configs easier
        if type(config) == dict:
            config = OmegaConf.create(config)

        self.config = config

        self.absorb = config.graph.type == "absorb"
        vocab_size = config.tokens + (1 if self.absorb else 0)
        num_classes = 5  # Updated for 5 cell types
        class_dropout_prob = 0.1

        self.vocab_embed = EmbeddingLayer(config.model.hidden_size, vocab_size)
        self.sigma_map = TimestepEmbedder(config.model.cond_dim)
        self.label_embed = LabelEmbedder(
            num_classes, config.model.cond_dim, class_dropout_prob
        )
        self.rotary_emb = rotary.Rotary(
            config.model.hidden_size // config.model.n_heads
        )

        self.blocks = nn.ModuleList(
            [
                DDiTBlock(
                    config.model.hidden_size,
                    config.model.n_heads,
                    config.model.cond_dim,
                    dropout=config.model.dropout,
                )
                for _ in range(config.model.n_blocks)
            ]
        )

        self.output_layer = DDitFinalLayer(
            config.model.hidden_size, vocab_size, config.model.cond_dim
        )
        self.scale_by_sigma = config.model.scale_by_sigma

        # label_emb is added for exp levels to match seq length i.e. deepstarr type datasets
        self.label_emb = nn.Sequential(
            nn.Linear(
                2, config.model.length
            ),  # change to 2 for transformer model inference
            nn.SiLU(),
            nn.Linear(config.model.length, config.model.length),
        )

        # Flag to enable attention score collection
        self.save_attention = False

    def _get_bias_dropout_scale(self):
        return (
            bias_dropout_add_scale_fused_train
            if self.training
            else bias_dropout_add_scale_fused_inference
        )

    def enable_attention_scoring(self, enable=True):
        """Enable or disable attention score collection."""
        self.save_attention = enable
        for block in self.blocks:
            block.save_attention = enable

    def get_attention_scores(self):
        """Get attention scores from all blocks."""
        # Only return attention scores from the last block
        if (
            hasattr(self.blocks[-1], "attention_scores")
            and self.blocks[-1].attention_scores is not None
        ):
            # Get attention tensor from last block
            attention = self.blocks[-1].attention_scores

            # Extract only the last head if there are multiple heads
            # Shape is [batch_size, n_heads, seq_len, seq_len]
            if attention.size(1) > 1:
                last_head_attention = attention[
                    :, -1:, :, :
                ]  # Keep the head dimension but only last head
            else:
                last_head_attention = attention

            return [last_head_attention]
        return []

    def forward(self, indices, labels, train, sigma):
        # Transformer-based forward pass only
        x = self.vocab_embed(indices, labels)
        c = F.silu(self.sigma_map(sigma))  # + self.label_embed(labels, train))
        rotary_cos_sin = self.rotary_emb(x)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, c, seqlens=None)

            x = self.output_layer(x, c)

        x = torch.scatter(x, -1, indices[..., None], torch.zeros_like(x[..., :1]))

        if self.save_attention:
            # Return both output and attention scores
            attention_scores = self.get_attention_scores()
            return x, attention_scores
        else:
            return x
