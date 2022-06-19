from typing import Callable, Sized, Union

from pytorch_lightning.trainer.connectors.data_connector import DataConnector
from torch.utils.data import Dataset, RandomSampler, SequentialSampler, BatchSampler, DataLoader
from torch_geometric.loader.base import BaseDataLoader


def _is_dataloader_shuffled(dataloader: DataLoader):
    return (
            hasattr(dataloader, "sampler")
            and not (  # Added this condition
        isinstance(dataloader.sampler, BatchSampler) and
        isinstance(dataloader.sampler.sampler, SequentialSampler),
    )
    )


# I am going insane from this warning. It's a false positive. Therefore we monkey patch it.
_check_eval_shuffling_og = DataConnector._check_eval_shuffling


def _check_eval_shuffling(cls, dataloader, mode):
    if not _is_dataloader_shuffled(dataloader):
        return

    _check_eval_shuffling_og(dataloader, mode)


DataConnector._check_eval_shuffling = _check_eval_shuffling


class BatchedLoader(BaseDataLoader):
    def __init__(
            self,
            dataset: Union[Sized, Dataset],
            transform: Callable = None,
            shuffle: bool = False,
            generator=None,
            batch_size: int = 1,
            drop_last: bool = False,
            batch_size_tmp: int = None,
            **kwargs,
    ):
        kwargs.pop('collate_fn', None)
        kwargs.pop('sampler', None)
        batch_size = batch_size or batch_size_tmp
        self.transform = transform
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.batch_size_tmp = batch_size

        # Default sampler to set autocollate to False
        if shuffle:
            sampler = RandomSampler(dataset, generator=generator)
        else:
            sampler = SequentialSampler(dataset)
        batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        super().__init__(
            dataset,
            collate_fn=self.sample,
            sampler=batch_sampler,
            batch_size=None,
            **kwargs,
        )

    def sample(self, inputs):
        return inputs

    def transform_fn(self, out):
        return out if self.transform is None else self.transform(out)
