import os

import wandb

from shared.config import ConnectionConfig
from shared.constants import BENCHMARKS_LOGS


def train():
    with wandb.init() as run:
        config = wandb.config
        for epoch in range(config["epochs"]):
            wandb.log({"loss": 0, "epoch": epoch})


def run():
    # wandb.init(
    #     # mode='offline',
    #     # dir=BENCHMARKS_LOGS,
    #     project='Thesis',
    #     # **kwargs
    # )

    sweep_config = {
        "name": "my-sweep",
        "method": "random",
        "parameters": {
            "epochs": {
                "values": [10, 20, 50]
            },
            "learning_rate": {
                "min": 0.0001,
                "max": 0.1
            },
            "test": {
                "value": 3
            }
        }
    }

    os.environ['WANDB_DIR'] = str(BENCHMARKS_LOGS)
    sweep_id = wandb.sweep(sweep_config)

    count = 5  # number of runs to execute
    wandb.agent(sweep_id, function=train, count=count)


if __name__ == "__main__":
    run()
