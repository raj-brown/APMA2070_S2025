class Inference:
    """Base class for all inference methods."""

    def __init__(self):
        self._params = None
        self.sampler = None

    @property
    def params(self):
        return self._params

    def make_sampler(self):
        """Creates a sampler for the inference."""
        raise NotImplementedError("Make sampler method to be implemented.")

    def sampling(self, sess):
        """Performs sampling."""
        raise NotImplementedError("Sampling method to be implemented.")
