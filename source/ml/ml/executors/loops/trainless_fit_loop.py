from pytorch_lightning.loops import FitLoop


class TrainlessFitLoop(FitLoop):
    def do_advance_loop(self):
        self.on_advance_start()
        self.on_advance_end()

    def run(self):
        if self.skip:
            return self.on_skip()

        self.reset()
        self.on_run_start()

        try:
            self.do_advance_loop()
        except StopIteration:
            pass

        output = self.on_run_end()
        return output

    def advance(self):
        """Advance from one iteration to the next."""
        pass

    def on_advance_end(self) -> None:
        super().on_advance_end()
        self.trainer._call_callback_hooks("on_validation_epoch_end")
