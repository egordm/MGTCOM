class BurnInMonitor:
    last_loss: float = float('inf')
    counter: int = 0
    burned_in: bool = False

    def __init__(self, patience: int = 2, threshold=0.01) -> None:
        super().__init__()
        self.patience = patience
        self.threshold = threshold

    def reset(self) -> None:
        self.last_loss = float('inf')
        self.counter = 0
        self.burned_in = False

    def update(self, loss: float) -> bool:
        # if loss >= self.last_loss * (1 - self.threshold):
        if loss <= self.last_loss * (1 + self.threshold):
            self.counter += 1
        else:
            self.counter = 0

        if self.counter >= self.patience:
            self.burned_in = True

        self.last_loss = loss
        return self.burned_in

