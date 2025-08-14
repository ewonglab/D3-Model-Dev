import abc
import torch
import torch.nn.functional as F
from src.utils.catsample import sample_categorical
import numpy as np

from src.models import model_utils as mutils

_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(f"Already registered model with name: {local_name}")
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_predictor(name):
    return _PREDICTORS[name]


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, graph, noise):
        super().__init__()
        self.graph = graph
        self.noise = noise

    @abc.abstractmethod
    def update_fn(self, score_fn, x, labels, t, step_size):
        """One update of the predictor.

        Args:
            score_fn: score function
            x: A PyTorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.

        Returns:
            x: A PyTorch tensor of the next state.
        """
        pass


@register_predictor(name="euler")
class EulerPredictor(Predictor):
    def update_fn(self, score_fn, x, labels, t, step_size):
        sigma, dsigma = self.noise(t)
        score_output = score_fn(x, labels, sigma)

        # Handle tuple return (score, attention_scores)
        if isinstance(score_output, tuple):
            score, _ = score_output  # Unpack score and discard attention for now
        else:
            score = score_output

        rev_rate = step_size * dsigma[..., None] * self.graph.reverse_rate(x, score)
        x = self.graph.sample_rate(x, rev_rate)
        return x


@register_predictor(name="none")
class NonePredictor(Predictor):
    def update_fn(self, score_fn, x, labels, t, step_size):
        return x


@register_predictor(name="analytic")
class AnalyticPredictor(Predictor):
    def update_fn(self, score_fn, x, labels, t, step_size):
        curr_sigma = self.noise(t)[0]
        next_sigma = self.noise(t - step_size)[0]
        dsigma = curr_sigma - next_sigma

        score_output = score_fn(x, labels, curr_sigma)

        # Handle tuple return (score, attention_scores)
        if isinstance(score_output, tuple):
            score, _ = score_output  # Unpack score and discard attention for now
        else:
            score = score_output

        stag_score = self.graph.staggered_score(score, dsigma)
        # print (stag_score.shape)
        probs = stag_score * self.graph.transp_transition(x, dsigma)
        return sample_categorical(probs)


class Denoiser:
    def __init__(self, graph, noise):
        self.graph = graph
        self.noise = noise

    def update_fn(self, score_fn, x, labels, t):
        sigma = self.noise(t)[0]

        score_output = score_fn(x, labels, sigma)

        # Handle tuple return (score, attention_scores)
        if isinstance(score_output, tuple):
            score, _ = score_output  # Unpack score and discard attention for now
        else:
            score = score_output

        stag_score = self.graph.staggered_score(score, sigma)
        probs = stag_score * self.graph.transp_transition(x, sigma)
        # truncate probabilities
        if self.graph.absorb:
            probs = probs[..., :-1]

        # return probs.argmax(dim=-1)
        return sample_categorical(probs)


def get_sampling_fn(config, graph, noise, batch_dims, eps, device):

    sampling_fn = get_pc_sampler(
        graph=graph,
        noise=noise,
        batch_dims=batch_dims,
        predictor=config.sampling.predictor,
        steps=config.sampling.steps,
        denoise=config.sampling.noise_removal,
        eps=eps,
        device=device,
    )

    return sampling_fn


def get_pc_sampler(
    graph,
    noise,
    batch_dims,
    predictor,
    steps,
    denoise=True,
    eps=1e-5,
    device=torch.device("cpu"),
    proj_fun=lambda x: x,
    save_attention=False,
):
    predictor = get_predictor(predictor)(graph, noise)
    projector = proj_fun
    denoiser = Denoiser(graph, noise)

    @torch.no_grad()
    def pc_sampler(model, labels, current_seed=None):
        # Enable attention scoring if requested, but only for the final step
        # We'll enable it just before the final step to save memory
        if save_attention and hasattr(model, "enable_attention_scoring"):
            # Don't enable it at the start - we'll enable it only for the final step
            model.enable_attention_scoring(False)

        sampling_score_fn = mutils.get_score_fn(model, train=False, sampling=True)
        x = graph.sample_limit(*batch_dims).to(device)
        timesteps = torch.linspace(1, eps, steps + 1, device=device)
        dt = (1 - eps) / steps

        # To store attention scores from final step only
        all_attention_scores = [] if save_attention else None

        # Set random seed for this batch if provided
        if current_seed is not None:
            # Convert seed to integer and set it
            seed = int(current_seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        for i in range(steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            x = projector(x)

            # Only enable attention scoring for the final iteration
            if (
                save_attention
                and i == steps - 1
                and hasattr(model, "enable_attention_scoring")
            ):
                model.enable_attention_scoring(True)

            # Call the predictor update function
            x = predictor.update_fn(sampling_score_fn, x, labels, t, dt)

            # Only collect attention scores at the final step
            if (
                save_attention
                and i == steps - 1
                and hasattr(model, "get_attention_scores")
            ):
                step_attention_scores = model.get_attention_scores()
                if step_attention_scores:
                    all_attention_scores.append((i, step_attention_scores))

        if denoise:
            # Enable attention scoring for final denoising step if needed
            if save_attention and hasattr(model, "enable_attention_scoring"):
                model.enable_attention_scoring(True)

            # denoising step
            x = projector(x)
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
            x = denoiser.update_fn(sampling_score_fn, x, labels, t)

            # Get final attention scores from denoising step
            if save_attention and hasattr(model, "get_attention_scores"):
                final_attention_scores = model.get_attention_scores()
                if final_attention_scores:
                    all_attention_scores.append(("final", final_attention_scores))

        # Disable attention scoring to save memory
        if save_attention and hasattr(model, "enable_attention_scoring"):
            model.enable_attention_scoring(False)

        # Return both generated sequence and attention scores
        if save_attention:
            return x, all_attention_scores
        else:
            return x

    return pc_sampler
