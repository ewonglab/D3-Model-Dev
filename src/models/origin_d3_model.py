import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from sklearn.decomposition import PCA
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

from einops import rearrange, einsum
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func

# from flash_attn.ops.fused_dense import FusedMLP, FusedDense
# from huggingface_hub import PyTorchModelHubMixin
from omegaconf import OmegaConf

from . import rotary
from .fused_add_dropout_scale import (
    bias_dropout_add_scale_fused_train,
    bias_dropout_add_scale_fused_inference,
    get_bias_dropout_add_scale,
    modulate_fused,
)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


@dataclass
class ConceptAttentionOutput:
    """Output structure for concept attention analysis"""
    sequences: torch.Tensor
    concept_attention_maps: Dict[str, torch.Tensor]
    cell_type_concepts: List[str]
    attention_weights: torch.Tensor


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


class CellTypeConceptDDiTBlock(nn.Module):
    """
    Enhanced DDiTBlock with Cell-Type ConceptAttention for DNA sequence generation.

    Updated version: Concept queries attend to DNA keys, allowing concepts to
    "look at" and attend to specific regions in the DNA sequence.

    This block enables the model to:
    1. Understand which DNA regions each cell-type concept focuses on
    2. Generate sequences with explicit cell-type regulatory patterns
    3. Visualize how cell type concepts attend to different DNA regions
    """

    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1,
                 num_cell_types=5, concept_attention_layers=None):
        super().__init__()

        # Original DDiTBlock components
        self.n_heads = n_heads
        self.dim = dim

        # DNA sequence attention components
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

        # ConceptAttention components for cell types
        self.concept_attention_layers = concept_attention_layers or list(range(15, 19))

        # DNA key/value projections for concept attention
        self.dna_proj_k = nn.Linear(dim, dim, bias=False)
        self.dna_proj_v = nn.Linear(dim, dim, bias=False)

        # Concept query projection
        self.concept_proj_q = nn.Linear(dim, dim, bias=False)

        # Normalization layers
        self.concept_norm = LayerNorm(dim)
        self.dna_norm = LayerNorm(dim)

        # Output projection for concept-attended features
        self.concept_out_proj = nn.Linear(dim, dim, bias=False)

        # Storage for concept attention visualization
        self.concept_attention_storage = {}

        # For storing attention scores (original functionality)
        self.attention_scores = None
        self.save_attention = False

    def store_concept_attention(self, attention_data: Dict[str, torch.Tensor]):
        """Store concept attention data for visualization"""
        self.concept_attention_storage.update(attention_data)

    def get_concept_attention(self) -> Dict[str, torch.Tensor]:
        """Retrieve stored concept attention data"""
        return self.concept_attention_storage

    def _get_bias_dropout_scale(self):
        """Same as original DDiTBlock"""
        return (
            bias_dropout_add_scale_fused_train
            if self.training
            else bias_dropout_add_scale_fused_inference
        )

    def forward(self, x, rotary_cos_sin, c, cell_type_concepts=None,
                layer_idx=None, seqlens=None):
        """
        Forward pass with cell-type concept attention - PRESERVES ORIGINAL GENERATION

        Updated: Concept queries attend to DNA keys to identify relevant DNA regions

        Args:
            x: DNA sequence embeddings [batch, seq_len, dim]
            rotary_cos_sin: Rotary position embeddings
            c: Conditioning vector from timestep/label embeddings
            cell_type_concepts: Cell type concept embeddings [batch, num_concepts, dim]
            layer_idx: Current layer index for selective concept attention
            seqlens: Sequence lengths for variable length sequences
        """

        # Use default concept embeddings if not provided
        if cell_type_concepts is None and hasattr(self, 'default_cell_type_concepts'):
            cell_type_concepts = self.default_cell_type_concepts

        # Use default layer index if not provided
        if layer_idx is None and hasattr(self, 'default_layer_idx'):
            layer_idx = self.default_layer_idx

        batch_size, seq_len = x.shape[0], x.shape[1]
        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
        )

        # ===== EXACT SAME DNA ATTENTION AS ORIGINAL DDiTBlock =====
        x_skip = x
        x = modulate_fused(self.norm1(x), shift_msa, scale_msa)
        qkv = self.attn_qkv(x)
        qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=self.n_heads)

        with torch.cuda.amp.autocast(enabled=False):
            cos, sin = rotary_cos_sin
            qkv = rotary.apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))

        # ===== UPDATED CONCEPT ATTENTION: CONCEPT QUERIES -> DNA KEYS =====
        q, k, v = qkv.unbind(dim=2)
        concept_attention_weights = None
        concept_attended_features = None

        if (cell_type_concepts is not None and
            layer_idx is not None and
            layer_idx in self.concept_attention_layers):

            # Project DNA sequences as keys and values for concept attention
            dna_k = self.dna_proj_k(self.dna_norm(x))  # [batch, seq_len, dim]
            dna_v = self.dna_proj_v(self.dna_norm(x))  # [batch, seq_len, dim]

            # Project cell type concepts as queries
            concept_q = self.concept_proj_q(self.concept_norm(cell_type_concepts))  # [batch, num_concepts, dim]

            # Reshape for multi-head attention
            dna_k = rearrange(dna_k, "b s (h d) -> b h s d", h=self.n_heads)
            dna_v = rearrange(dna_v, "b s (h d) -> b h s d", h=self.n_heads)
            concept_q = rearrange(concept_q, "b c (h d) -> b h c d", h=self.n_heads)

            # Cross-attention: Concept queries attend to DNA keys
            scale = (concept_q.shape[-1]) ** -0.5
            concept_attention_weights = torch.einsum('b h c d, b h s d -> b h c s',
                                                   concept_q, dna_k) * scale
            concept_attention_weights = F.softmax(concept_attention_weights, dim=-1)

            # Compute concept-attended DNA features
            concept_attended_features = torch.einsum('b h c s, b h s d -> b h c d',
                                                   concept_attention_weights, dna_v)
            concept_attended_features = rearrange(concept_attended_features, "b h c d -> b c (h d)")

            # Project back to original dimension
            concept_attended_features = self.concept_out_proj(concept_attended_features)

            # Store for visualization (doesn't affect generation)
            self.store_concept_attention({
                "concept_queries": concept_q,
                "dna_keys": dna_k,
                "dna_values": dna_v,
                "concept_attention_weights": concept_attention_weights.detach().cpu(),
                "concept_attended_features": concept_attended_features.detach().cpu(),
                "layer_idx": layer_idx
            })

        # ===== EXACT SAME FLASH ATTENTION AS ORIGINAL =====
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

        # Compute attention scores if needed (original functionality)
        if self.save_attention:
            # Need to properly reshape qkv back to batch format first
            qkv_reshaped = rearrange(qkv, "(b s) ... -> b s ...", b=batch_size)

            # Extract q, k, v from reshaped tensor
            q_batch = qkv_reshaped[:, :, 0]  # Shape: [batch_size, seq_len, n_heads, dim]
            k_batch = qkv_reshaped[:, :, 1]  # Shape: [batch_size, seq_len, n_heads, dim]

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
                        q_head.view(batch_size, seq_len, -1),  # [batch_size, seq_len, dim]
                        k_head.view(batch_size, seq_len, -1).transpose(1, 2),  # [batch_size, dim, seq_len]
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

        # ===== EXACT SAME RESIDUAL AND MLP AS ORIGINAL =====
        x = bias_dropout_scale_fn(
            self.attn_out(x), None, gate_msa, x_skip, self.dropout
        )

        # MLP block (exact same as original)
        x = bias_dropout_scale_fn(
            self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)),
            None,
            gate_mlp,
            x,
            self.dropout,
        )

        # Note: concept_attended_features are computed for analysis but don't affect the main generation path
        # They could be used for additional conditioning or analysis in future versions

        return x


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

        n = 256
        embed_dim = 256
        # self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
        #                            nn.Linear(embed_dim, embed_dim))
        self.linear = nn.Conv1d(vocab_size + 1, n, kernel_size=9, padding=4)
        # self.linear = nn.Conv1d(vocab_size + 1, n, kernel_size=7, padding='same')
        self.conv_blocks = nn.ModuleList(
            [
                nn.Conv1d(n, n, kernel_size=9, padding=4),
                nn.Conv1d(n, n, kernel_size=9, padding=4),
                nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
                nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
                nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256),
                nn.Conv1d(n, n, kernel_size=9, padding=4),
                nn.Conv1d(n, n, kernel_size=9, padding=4),
                nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
                nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
                nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256),
                nn.Conv1d(n, n, kernel_size=9, padding=4),
                nn.Conv1d(n, n, kernel_size=9, padding=4),
                nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
                nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
                nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256),
                nn.Conv1d(n, n, kernel_size=9, padding=4),
                nn.Conv1d(n, n, kernel_size=9, padding=4),
                nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
                nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
                nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256),
            ]
        )

        # self.conv_blocks = nn.ModuleList([nn.Conv1d(n, n, kernel_size=3, padding='same'),
        #                                   nn.Conv1d(n, n, kernel_size=5, padding='same'),
        #                                   nn.Conv1d(n, n, kernel_size=3, padding='same')
        #                                   ])
        self.denses = nn.ModuleList([Dense(embed_dim, n) for _ in range(20)])
        self.norms = nn.ModuleList([nn.GroupNorm(1, n) for _ in range(20)])

        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.scale = nn.Parameter(torch.ones(1))
        self.final = nn.Sequential(
            nn.Conv1d(n, n, kernel_size=1), nn.GELU(), nn.Conv1d(n, 4, kernel_size=1)
        )
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
        self.attention_layer_idx = -1  # Default to last layer

        # PCA-based feature extraction flags
        self.save_pca_features = False
        self.layer_hidden_states = []  # Store intermediate layer outputs
        self.pca_components = 5  # Number of PCA components to keep
        self.pca_models = {}  # Store fitted PCA models for each position

        # Concept attention flags and storage
        self.save_concept_attention = False
        self.concept_attention_layers = []  # Which layers to apply concept attention
        self.concept_attention_storage = {}  # Store concept attention maps
        self.cell_type_names = {
            0: "Endothelial",
            1: "Fibroblast",
            2: "Smooth_Muscle",
            3: "Ventricular_Cardiomyocyte",
            4: "Atrial_Cardiomyocyte"
        }

    def _get_bias_dropout_scale(self):
        return (
            bias_dropout_add_scale_fused_train
            if self.training
            else bias_dropout_add_scale_fused_inference
        )

    def enable_attention_scoring(self, enable=True, layer_idx=None):
        """
        Enable or disable attention score collection.
        
        Args:
            enable: Whether to enable attention scoring
            layer_idx: Which attention layer to extract scores from (0-indexed)
                      If None, uses the previously set layer_idx
        """
        self.save_attention = enable
        
        # Update layer index if provided
        if layer_idx is not None:
            # Convert to integer and validate
            try:
                layer_idx = int(layer_idx)
                if layer_idx >= len(self.blocks):
                    print(f"Warning: Layer index {layer_idx} out of range. Using last layer.")
                    layer_idx = -1
                elif layer_idx < 0:
                    print(f"Warning: Negative layer index {layer_idx}. Using last layer.")
                    layer_idx = -1
                self.attention_layer_idx = layer_idx
            except ValueError:
                print(f"Warning: Invalid layer index {layer_idx}. Using last layer.")
                self.attention_layer_idx = -1
        
        # Enable attention scoring for all blocks or just the specified one
        for i, block in enumerate(self.blocks):
            # If we're targeting a specific layer, only enable for that one
            if self.attention_layer_idx >= 0:
                block.save_attention = enable and (i == self.attention_layer_idx)
            else:
                block.save_attention = enable

    def enable_pca_features(self, enable=True, n_components=5):
        """
        Enable or disable PCA-based feature extraction from layer outputs.
        
        Args:
            enable: Whether to enable PCA feature extraction
            n_components: Number of PCA components to keep
        """
        self.save_pca_features = enable
        self.pca_components = n_components
        if enable:
            self.layer_hidden_states = []
            self.pca_models = {}
        
    def compute_pca_scores(self):
        """
        Compute PCA scores from collected layer hidden states.
        
        Returns:
            pca_scores: Tensor of shape [batch_size, seq_len, n_components]
        """
        if not self.layer_hidden_states:
            return None
            
        # Stack all layer outputs: [n_layers, batch_size, seq_len, hidden_dim]
        stacked_layers = torch.stack(self.layer_hidden_states, dim=0)
        n_layers, batch_size, seq_len, hidden_dim = stacked_layers.shape
        
        # Reshape to [batch_size, seq_len, n_layers * hidden_dim]
        # This creates a feature vector for each position that contains info from all layers
        layer_features = stacked_layers.permute(1, 2, 0, 3).reshape(batch_size, seq_len, -1)
        
        # Convert to numpy for sklearn PCA
        layer_features_np = layer_features.detach().cpu().numpy()
        
        # Compute PCA for each sequence position across the batch
        pca_scores = torch.zeros(batch_size, seq_len, self.pca_components, device=stacked_layers.device)
        
        for pos in range(seq_len):
            # Get features for this position across all samples: [batch_size, n_layers * hidden_dim]
            pos_features = layer_features_np[:, pos, :]
            
            # Skip if we have constant features (no variance)
            if np.var(pos_features) < 1e-8:
                continue
                
            # Fit PCA for this position
            pca = PCA(n_components=min(self.pca_components, pos_features.shape[0]-1, pos_features.shape[1]))
            try:
                pca_transformed = pca.fit_transform(pos_features)
                
                # Pad with zeros if we have fewer components than requested
                if pca_transformed.shape[1] < self.pca_components:
                    padded = np.zeros((batch_size, self.pca_components))
                    padded[:, :pca_transformed.shape[1]] = pca_transformed
                    pca_transformed = padded
                    
                pca_scores[:, pos, :] = torch.from_numpy(pca_transformed).to(stacked_layers.device)
                
                # Store the fitted PCA model for potential future use
                self.pca_models[pos] = pca
                
            except Exception as e:
                print(f"PCA failed for position {pos}: {e}")
                continue
        
        return pca_scores

    def get_pca_scores(self):
        """Get PCA scores computed from layer hidden states."""
        if self.save_pca_features:
            return self.compute_pca_scores()
        return None

    def enable_concept_attention(self, enable=True, layers=None):
        """
        Enable or disable concept attention for specified layers.

        Args:
            enable: Whether to enable concept attention
            layers: List of layer indices to replace with concept attention blocks.
                   If None, uses default layers [15, 16, 17, 18]
        """
        self.save_concept_attention = enable

        if layers is not None:
            self.concept_attention_layers = layers
        elif not self.concept_attention_layers:
            # Default to last few layers
            total_layers = len(self.blocks)
            self.concept_attention_layers = list(range(max(0, total_layers-4), total_layers))

        if enable:
            self._replace_blocks_with_concept_attention()
        else:
            # Clear concept attention storage
            self.concept_attention_storage = {}

    def _replace_blocks_with_concept_attention(self):
        """Replace specified DDiTBlocks with CellTypeConceptDDiTBlocks"""
        replaced_count = 0

        for layer_idx in self.concept_attention_layers:
            if layer_idx < len(self.blocks):
                original_block = self.blocks[layer_idx]

                # Check if it's a DDiTBlock and not already a concept block
                if (hasattr(original_block, 'norm1') and
                    hasattr(original_block, 'n_heads') and
                    not isinstance(original_block, CellTypeConceptDDiTBlock)):

                    # Create concept attention block with same parameters
                    concept_block = CellTypeConceptDDiTBlock(
                        dim=original_block.norm1.dim,
                        n_heads=original_block.n_heads,
                        cond_dim=original_block.adaLN_modulation.in_features,
                        mlp_ratio=4,
                        dropout=original_block.dropout,
                        num_cell_types=5,
                        concept_attention_layers=self.concept_attention_layers
                    ).to(next(original_block.parameters()).device)

                    # Copy weights from original block
                    self._copy_block_weights(original_block, concept_block)

                    # Replace the block
                    self.blocks[layer_idx] = concept_block
                    replaced_count += 1

        print(f"✅ Replaced {replaced_count} blocks with ConceptAttention")

    def _copy_block_weights(self, original_block, concept_block):
        """Copy weights from original DDiTBlock to CellTypeConceptDDiTBlock"""
        try:
            # Copy standard attention weights
            concept_block.norm1.weight.data = original_block.norm1.weight.data.clone()
            concept_block.attn_qkv.weight.data = original_block.attn_qkv.weight.data.clone()
            concept_block.attn_out.weight.data = original_block.attn_out.weight.data.clone()

            # Copy MLP weights
            concept_block.norm2.weight.data = original_block.norm2.weight.data.clone()
            concept_block.mlp[0].weight.data = original_block.mlp[0].weight.data.clone()
            concept_block.mlp[0].bias.data = original_block.mlp[0].bias.data.clone()
            concept_block.mlp[2].weight.data = original_block.mlp[2].weight.data.clone()
            concept_block.mlp[2].bias.data = original_block.mlp[2].bias.data.clone()

            # Copy AdaLN modulation weights
            concept_block.adaLN_modulation.weight.data = original_block.adaLN_modulation.weight.data.clone()
            concept_block.adaLN_modulation.bias.data = original_block.adaLN_modulation.bias.data.clone()

            # Initialize concept projection layers with small random values
            nn.init.xavier_uniform_(concept_block.dna_proj_k.weight, gain=0.1)
            nn.init.xavier_uniform_(concept_block.dna_proj_v.weight, gain=0.1)
            nn.init.xavier_uniform_(concept_block.concept_proj_q.weight, gain=0.1)
            nn.init.xavier_uniform_(concept_block.concept_out_proj.weight, gain=0.1)

        except Exception as e:
            print(f"⚠️ Warning: Could not copy all weights: {e}")

    def _create_concept_embeddings(self, expressions: torch.Tensor) -> torch.Tensor:
        """Create cell type concept embeddings from expression data"""
        batch_size = expressions.shape[0]

        # Get model's hidden dimension from the first block
        model_dim = self.blocks[0].norm1.dim

        # Create concept embeddings for each cell type in the batch
        concept_embeddings = []

        # For each sample in batch, create embeddings for all 5 cell types
        for i in range(batch_size):
            sample_concepts = []

            # Create embeddings for all 5 cell types as concepts
            for cell_type_idx in range(5):
                embed = self.label_embed(torch.tensor([cell_type_idx], device=expressions.device), train=False)
                embed = embed.squeeze(0)  # Remove batch dimension

                # Project to model dimension if needed
                if embed.shape[-1] != model_dim:
                    # Create a simple linear projection
                    if not hasattr(self, '_concept_projection'):
                        self._concept_projection = nn.Linear(embed.shape[-1], model_dim).to(expressions.device)
                    embed = self._concept_projection(embed)

                sample_concepts.append(embed)

            # Stack concepts for this sample: [5, model_dim]
            sample_concept_stack = torch.stack(sample_concepts, dim=0)
            concept_embeddings.append(sample_concept_stack)

        # Stack all samples: [batch, 5, model_dim]
        cell_type_concepts = torch.stack(concept_embeddings, dim=0)

        return cell_type_concepts

    def _enable_concept_forward(self, cell_type_concepts: torch.Tensor):
        """Modify model blocks to use ConceptAttention during forward pass"""

        # Store concept embeddings directly in each ConceptAttention block as attributes
        for layer_idx, block in enumerate(self.blocks):
            if isinstance(block, CellTypeConceptDDiTBlock):
                # Set concept embeddings and layer index as attributes
                block.default_cell_type_concepts = cell_type_concepts
                block.default_layer_idx = layer_idx

    def get_concept_attention_scores(self):
        """Get concept attention scores from all ConceptAttention blocks"""
        concept_attention_maps = {}

        for layer_idx, block in enumerate(self.blocks):
            if isinstance(block, CellTypeConceptDDiTBlock):
                block_attention = block.get_concept_attention()
                if block_attention:
                    concept_attention_maps[f'layer_{layer_idx}'] = block_attention

        return concept_attention_maps

    def extract_concept_attention_for_sequence(self, concept_attention_maps: Dict[str, Any],
                                              seq_idx: int, target_cell_type: str,
                                              attention_type: str = 'specific') -> Dict[str, List[float]]:
        """Extract ConceptAttention scores for a specific sequence

        Args:
            concept_attention_maps: Attention maps from model
            seq_idx: Index of sequence in batch
            target_cell_type: Target cell type name
            attention_type: 'specific' for target cell type only, 'all' for all 5 cell types

        Returns:
            Dict with cell type names as keys and attention scores as values
        """

        # Map cell type name to concept index
        cell_type_to_idx = {
            "atrial_cardiomyocyte": 4,
            "endothelial": 0,
            "fibroblast": 1,
            "smooth_muscle": 2,
            "ventricular_cardiomyocyte": 3
        }

        idx_to_cell_type = {v: k for k, v in cell_type_to_idx.items()}

        target_concept_idx = cell_type_to_idx.get(target_cell_type.lower())
        if target_concept_idx is None:
            print(f"⚠️ Unknown cell type: {target_cell_type}")
            return {}

        # Input validation and conversion
        if not concept_attention_maps:
            print(f"⚠️ No concept attention maps provided")
            return {}

        # Handle both dict and list inputs
        if isinstance(concept_attention_maps, list):
            # If it's a list, it might be a list of concept attention dicts from different steps
            # Try to find the last non-empty dict in the list
            for item in reversed(concept_attention_maps):
                if isinstance(item, dict) and item:
                    concept_attention_maps = item
                    break
            else:
                # No valid dict found in list, return empty result silently
                return {}
        elif not isinstance(concept_attention_maps, dict):
            print(f"⚠️ Expected concept_attention_maps to be dict or list, got {type(concept_attention_maps)}")
            return {}

        # Find the last layer's attention data (should be the most informative)
        for layer_name, attention_data in reversed(list(concept_attention_maps.items())):
            if "concept_attention_weights" in attention_data:
                # Updated attention shape: [batch, heads, num_concepts, seq_len]
                # (concept queries attending to DNA keys)
                weights = attention_data["concept_attention_weights"]
                if seq_idx >= weights.shape[0]:
                    print(f"⚠️ Sequence index {seq_idx} out of range for batch size {weights.shape[0]}")
                    continue

                result = {}

                if attention_type == 'specific':
                    # Extract attention for target cell type only: [heads, target_concept, seq_len]
                    target_weights = weights[seq_idx, :, target_concept_idx, :]  # [heads, seq_len]
                    # Average over heads to get [seq_len] attention scores for target cell type
                    seq_weights = target_weights.mean(dim=0)  # [seq_len]
                    result[target_cell_type.lower()] = seq_weights.numpy().tolist()

                elif attention_type == 'all':
                    # Extract attention for all 5 cell types
                    for concept_idx in range(5):
                        cell_type_name = idx_to_cell_type[concept_idx]
                        concept_weights = weights[seq_idx, :, concept_idx, :]  # [heads, seq_len]
                        # Average over heads to get [seq_len] attention scores
                        seq_weights = concept_weights.mean(dim=0)  # [seq_len]
                        result[cell_type_name] = seq_weights.numpy().tolist()

                else:
                    print(f"⚠️ Unknown attention_type: {attention_type}. Use 'specific' or 'all'")
                    return {}

                return result

        return {}  # Return empty dict if no attention found

    def get_attention_scores(self):
        """Get attention scores from the specified block."""
        # Determine which block to get scores from
        target_idx = self.attention_layer_idx
        if target_idx < 0:  # Negative index means count from the end
            target_idx = len(self.blocks) + target_idx
        
        # Ensure index is valid
        if target_idx < 0 or target_idx >= len(self.blocks):
            return []
            
        # Get attention scores from the target block
        if (
            hasattr(self.blocks[target_idx], "attention_scores")
            and self.blocks[target_idx].attention_scores is not None
        ):
            # Get attention tensor from target block
            attention = self.blocks[target_idx].attention_scores

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
        # ---------------------------------------------#
        # Below code for transformer based networks
        x = self.vocab_embed(indices, labels)
        c = F.silu(self.sigma_map(sigma))  # + self.label_embed(labels, train))
        rotary_cos_sin = self.rotary_emb(x)

        # Clear previous layer states if collecting PCA features
        if self.save_pca_features:
            self.layer_hidden_states = []

        # Clear concept attention storage
        if self.save_concept_attention:
            self.concept_attention_storage = {}

        # Create concept embeddings if concept attention is enabled
        cell_type_concepts = None
        if self.save_concept_attention:
            cell_type_concepts = self._create_concept_embeddings(labels)
            self._enable_concept_forward(cell_type_concepts)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                # Check if this is a concept attention block
                if isinstance(self.blocks[i], CellTypeConceptDDiTBlock):
                    x = self.blocks[i](x, rotary_cos_sin, c,
                                     cell_type_concepts=cell_type_concepts,
                                     layer_idx=i, seqlens=None)
                else:
                    x = self.blocks[i](x, rotary_cos_sin, c, seqlens=None)

                # Store layer output for PCA analysis
                if self.save_pca_features:
                    # Convert to float32 and detach for PCA computation
                    self.layer_hidden_states.append(x.detach().float())

            x = self.output_layer(x, c)

        x = torch.scatter(x, -1, indices[..., None], torch.zeros_like(x[..., :1]))

        # Collect concept attention from all concept blocks
        if self.save_concept_attention:
            self.concept_attention_storage = self.get_concept_attention_scores()
        # ---------------------------------------------#

        # Comment out the above section and uncomment below for convolution based networks
        # x = torch.nn.functional.one_hot(indices, num_classes=4).float()
        # label = torch.unsqueeze(self.label_emb(labels), dim=2)
        # x = torch.cat([x, label], dim=-1)
        # x = x.permute(0, 2, 1)
        # out = self.act(self.linear(x))
        #
        # c = F.silu(self.sigma_map(sigma))  # + self.label_embed(labels, train))
        #
        # for block, dense, norm in zip(self.conv_blocks, self.denses, self.norms):
        #     h = self.act(block(norm(out + dense(c)[:, :, None])))
        #     if h.shape == out.shape:
        #         out = h + out
        #     else:
        #         out = h
        #
        # x = self.final(out)
        # x = x.permute(0, 2, 1)

        # ---------------------------------------------#

        if self.save_attention and self.save_concept_attention:
            # Return output, attention scores, and concept attention
            attention_scores = self.get_attention_scores()
            concept_attention = self.concept_attention_storage
            return x, attention_scores, concept_attention
        elif self.save_attention:
            # Return both output and attention scores
            attention_scores = self.get_attention_scores()
            return x, attention_scores
        elif self.save_concept_attention:
            # Return output and concept attention
            concept_attention = self.concept_attention_storage
            return x, concept_attention
        else:
            return x
