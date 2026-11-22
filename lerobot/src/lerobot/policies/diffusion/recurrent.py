"""Episode-local recurrent conditioning for offline evaluation."""


class EpisodeGRUContext:
    """Consume every real frame once, even when scoring sparse replanning anchors."""

    def __init__(self, model, state_at):
        self.model = model
        self.state_at = state_at
        self.frame = -1
        self.hidden = None
        self.condition = None

    def at(self, frame):
        if frame < 0 or frame < self.frame:
            raise ValueError("GRU episode frames must be requested in chronological order.")
        for index in range(self.frame + 1, frame + 1):
            self.condition, self.hidden = self.model.gru_step(self.state_at(index), self.hidden)
        self.frame = frame
        return self.condition
