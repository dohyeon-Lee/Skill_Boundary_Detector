"""Small skill-conditioned proprioceptive terminators."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class StateSkillMLPTerminator(nn.Module):
    """Predict a boundary from only the current proprio state and FSQ skill."""

    def __init__(
        self,
        *,
        state_dim: int,
        skill_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        termination_only: bool = True,
    ) -> None:
        super().__init__()
        if min(state_dim, skill_dim, hidden_dim, num_layers) <= 0:
            raise ValueError("State MLP dimensions and layer count must be positive.")
        self.state_dim = int(state_dim)
        self.skill_dim = int(skill_dim)
        self.termination_only = bool(termination_only)
        input_dim = self.state_dim + self.skill_dim
        layers: list[nn.Module] = [nn.LayerNorm(input_dim)]
        width = input_dim
        for _ in range(int(num_layers)):
            layers.extend((nn.Linear(width, hidden_dim), nn.SiLU()))
            width = hidden_dim
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)
        # Keep the original termination-only module and its state-dict keys
        # unchanged. The optional progress head exists only for joint training.
        if not self.termination_only:
            self.progress_head = nn.Linear(hidden_dim, 1)

    def _features(self, z_q: Tensor, state: Tensor) -> Tensor:
        """Validate inputs and return the last shared MLP representation."""
        if z_q.ndim != 2 or state.ndim != 2:
            raise ValueError(
                "State MLP expects z_q [B, skill_dim] and state [B, state_dim], "
                f"got {tuple(z_q.shape)} and {tuple(state.shape)}."
            )
        if z_q.shape[0] != state.shape[0]:
            raise ValueError("State MLP skill/state batch sizes do not match.")
        if z_q.shape[-1] != self.skill_dim or state.shape[-1] != self.state_dim:
            raise ValueError(
                "State MLP input dimensions do not match its checkpoint contract: "
                f"skill={z_q.shape[-1]}/{self.skill_dim}, "
                f"state={state.shape[-1]}/{self.state_dim}."
            )
        return self.network[:-1](torch.cat((state, z_q), dim=-1))

    def forward(self, z_q: Tensor, state: Tensor) -> Tensor:
        """Return the termination logit (the original inference contract)."""
        return self.network[-1](self._features(z_q, state)).squeeze(-1)

    def forward_outputs(self, z_q: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(progress, termination_logit)`` for the training objective."""
        features = self._features(z_q, state)
        termination_logits = self.network[-1](features).squeeze(-1)
        if self.termination_only:
            progress = torch.zeros_like(termination_logits).detach()
        else:
            progress = torch.sigmoid(self.progress_head(features)).squeeze(-1)
        return progress, termination_logits


class _CheckpointSafeRNN(nn.RNN):
    """Vanilla RNN without cuDNN's shared flat-weight storage.

    cuDNN's optional RNN weight packing turns each parameter into a view of a
    larger shared storage. ``safetensors.save_model`` rejects those views, so
    keep the ordinary per-parameter storages. The recurrent computation is
    still the standard ``nn.RNN`` implementation.
    """

    def flatten_parameters(self) -> None:
        return


class StateSkillRNNTerminator(nn.Module):
    """Predict a boundary from proprio history with a vanilla tanh RNN."""

    def __init__(
        self,
        *,
        state_dim: int,
        skill_dim: int,
        input_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        termination_only: bool = True,
    ) -> None:
        super().__init__()
        if min(state_dim, skill_dim, input_dim, hidden_dim, num_layers) <= 0:
            raise ValueError("State RNN dimensions and layer count must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("State RNN dropout must be in [0, 1).")
        self.state_dim = int(state_dim)
        self.skill_dim = int(skill_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.termination_only = bool(termination_only)
        combined_dim = self.state_dim + self.skill_dim
        self.input_encoder = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Linear(combined_dim, input_dim),
            nn.SiLU(),
        )
        self.rnn = _CheckpointSafeRNN(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            nonlinearity="tanh",
            batch_first=True,
            dropout=float(dropout) if self.num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(self.hidden_dim, 1)
        # As with the MLP, termination-only checkpoints retain exactly the
        # original parameter set and names.
        if not self.termination_only:
            self.progress_head = nn.Linear(self.hidden_dim, 1)

    def _sequence_inputs(self, z_q: Tensor, states: Tensor) -> Tensor:
        if states.ndim != 3:
            raise ValueError(
                "State RNN expects states [B, T, state_dim], "
                f"got {tuple(states.shape)}."
            )
        if z_q.ndim == 2:
            z_q = z_q[:, None, :].expand(-1, states.shape[1], -1)
        if z_q.ndim != 3 or z_q.shape[:2] != states.shape[:2]:
            raise ValueError(
                "State RNN expects z_q [B, skill_dim] or [B, T, skill_dim] "
                f"matching states, got {tuple(z_q.shape)}."
            )
        if z_q.shape[-1] != self.skill_dim or states.shape[-1] != self.state_dim:
            raise ValueError(
                "State RNN input dimensions do not match its checkpoint contract: "
                f"skill={z_q.shape[-1]}/{self.skill_dim}, "
                f"state={states.shape[-1]}/{self.state_dim}."
            )
        return self.input_encoder(torch.cat((states, z_q), dim=-1))

    @staticmethod
    def compact_valid_suffix(sequence: Tensor, lengths: Tensor) -> Tensor:
        """Move each valid suffix to the left for ``pack_padded_sequence``."""
        batch_size, sequence_length, width = sequence.shape
        positions = torch.arange(sequence_length, device=sequence.device)
        source = sequence_length - lengths[:, None] + positions[None, :]
        source = source.clamp(0, sequence_length - 1)
        compact = sequence.gather(1, source[..., None].expand(-1, -1, width))
        valid = positions[None, :] < lengths[:, None]
        return compact * valid[..., None].to(compact.dtype)

    def _sequence_hidden_states(
        self,
        z_q: Tensor,
        states: Tensor,
        *,
        lengths: Tensor | None = None,
        hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return every causal RNN output and the final hidden state."""
        encoded = self._sequence_inputs(z_q, states)
        if lengths is None:
            return self.rnn(encoded, hidden)

        lengths = lengths.to(device=encoded.device, dtype=torch.long).view(-1)
        if lengths.shape[0] != encoded.shape[0]:
            raise ValueError("State RNN sequence lengths do not match batch size.")
        lengths = lengths.clamp(1, encoded.shape[1])
        compact = self.compact_valid_suffix(encoded, lengths)
        packed = pack_padded_sequence(
            compact,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_outputs, next_hidden = self.rnn(packed, hidden)
        outputs, _ = pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=encoded.shape[1],
        )
        return outputs, next_hidden

    def _sequence_features(
        self,
        z_q: Tensor,
        states: Tensor,
        *,
        lengths: Tensor | None = None,
        hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return the final valid RNN feature and hidden state.

        ``states`` is chronologically ordered. When ``lengths`` is supplied, it
        describes a valid suffix (left padding is ignored), which lets training
        truncate history at the current skill boundary.
        """
        outputs, next_hidden = self._sequence_hidden_states(
            z_q,
            states,
            lengths=lengths,
            hidden=hidden,
        )
        if lengths is None:
            return outputs[:, -1], next_hidden
        return next_hidden[-1], next_hidden

    def forward_all_outputs(
        self,
        z_q: Tensor,
        states: Tensor,
        *,
        lengths: Tensor | None = None,
        hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return per-timestep ``(progress, termination_logit, hidden)``."""
        features, next_hidden = self._sequence_hidden_states(
            z_q,
            states,
            lengths=lengths,
            hidden=hidden,
        )
        termination_logits = self.head(features).squeeze(-1)
        if self.termination_only:
            progress = torch.zeros_like(termination_logits).detach()
        else:
            progress = torch.sigmoid(self.progress_head(features)).squeeze(-1)
        return progress, termination_logits, next_hidden

    def forward_sequence(
        self,
        z_q: Tensor,
        states: Tensor,
        *,
        lengths: Tensor | None = None,
        hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return the final-step termination logit and hidden state."""
        features, next_hidden = self._sequence_features(
            z_q,
            states,
            lengths=lengths,
            hidden=hidden,
        )
        return self.head(features).squeeze(-1), next_hidden

    def forward_sequence_outputs(
        self,
        z_q: Tensor,
        states: Tensor,
        *,
        lengths: Tensor | None = None,
        hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(progress, termination_logit, next_hidden)``."""
        features, next_hidden = self._sequence_features(
            z_q,
            states,
            lengths=lengths,
            hidden=hidden,
        )
        termination_logits = self.head(features).squeeze(-1)
        if self.termination_only:
            progress = torch.zeros_like(termination_logits).detach()
        else:
            progress = torch.sigmoid(self.progress_head(features)).squeeze(-1)
        return progress, termination_logits, next_hidden

    def step(
        self,
        z_q: Tensor,
        state: Tensor,
        hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Online inference: ``(state, skill, hidden) -> (logit, next_hidden)``."""
        if state.ndim != 2:
            raise ValueError(
                f"State RNN step expects state [B, D], got {tuple(state.shape)}."
            )
        return self.forward_sequence(
            z_q,
            state[:, None, :],
            hidden=hidden,
        )

    def step_outputs(
        self,
        z_q: Tensor,
        state: Tensor,
        hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Online ``(progress, termination_logit, next_hidden)`` inference."""
        if state.ndim != 2:
            raise ValueError(
                f"State RNN step expects state [B, D], got {tuple(state.shape)}."
            )
        return self.forward_sequence_outputs(
            z_q,
            state[:, None, :],
            hidden=hidden,
        )

    def initial_hidden(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        reference = next(self.parameters())
        return torch.zeros(
            self.num_layers,
            int(batch_size),
            self.hidden_dim,
            device=reference.device if device is None else device,
            dtype=reference.dtype if dtype is None else dtype,
        )
