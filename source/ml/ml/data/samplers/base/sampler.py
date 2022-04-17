from abc import abstractmethod


class Sampler:
    def __call__(self, *args, **kwargs):
        return self.sample(*args, **kwargs)

    @abstractmethod
    def sample(self, *args, **kwargs):
        raise NotImplementedError
