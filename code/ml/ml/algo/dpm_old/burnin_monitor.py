from ml.algo.dpm import Float


class BurnInMonitor:
    last_loss: Float
    counter: int
    burned_in: bool

    def __init__(self, patience: int = 2, threshold=0.01) -> None:
        super().__init__()
        self.patience = patience
        self.threshold = threshold

        self.reset()

    def reset(self) -> None:
        self.last_loss = 0
        self.counter = 0
        self.burned_in = False

    def update(self, loss: Float) -> bool:
        if loss <= self.last_loss * (1 + self.threshold):
            self.counter += 1

        if self.counter >= self.patience:
            self.burned_in = True

        self.last_loss = loss
        return self.burned_in
