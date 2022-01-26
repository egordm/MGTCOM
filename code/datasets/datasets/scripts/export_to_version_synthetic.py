import os
import shutil
from dataclasses import dataclass
from typing import Optional

from simple_parsing import field

from datasets.synthetic import generate_rdyn
from shared.cli import parse_args
from shared.logger import get_logger
from shared.schema import DatasetSchema, DatasetVersionType

LOG = get_logger(os.path.basename(__file__))


@dataclass
class Args:
    dataset: str = field(positional=True, help="Dataset name")
    version: Optional[str] = field(default=None, help="Dataset version")
    force: Optional[bool] = field(default=False, help="Force overwrite")
    save_graphml: Optional[bool] = field(default=True, help="Save graphml")


def run(args: Args):
    DATASET = DatasetSchema.load_schema(args.dataset)
    VERSION = DATASET.get_version(args.version)

    if not DATASET.is_synthetic():
        LOG.warning(f'Dataset {DATASET.name} is not synthetic, skipping')
        return

    if VERSION.get_path().exists() and not os.getenv('FORCE'):
        LOG.info(f"Dataset version {args.dataset}:{args.version} already exists. Use --force to overwrite.")
        return

    # Delete existing data
    if VERSION.get_path().exists():
        shutil.rmtree(VERSION.get_path())

    method = VERSION.get_param('method')
    if method == 'rdyn':
        snapshots = generate_rdyn(**VERSION.parameters)
    else:
        LOG.error(f'Method {method} not supported')
        exit(1)

    TRAIN_PART = VERSION.train
    TRAIN_PART.get_path().mkdir(parents=True, exist_ok=True)

    if VERSION.type == DatasetVersionType.EDGELIST_SNAPSHOTS:
        # Create snapshots
        for i, (G_snapshot, comms) in enumerate(snapshots):
            output_file = TRAIN_PART.snapshot_edgelist(i)
            LOG.info(f'Writing snapshot {i} to {output_file}')
            G_snapshot.save_edgelist(str(output_file))

            LOG.info(f'Saving ground-truth to {output_file.with_suffix(".comlist")}')
            comms_snapshot = comms.filter_nodes(G_snapshot.vs['gid'])
            comms_snapshot.save_comlist(str(TRAIN_PART.snapshot_ground_truth(i)))

            if args.save_graphml:
                LOG.info(f'Saving graphml to {output_file.with_suffix(".graphml")}')
                G_snapshot.write_graphml(str(output_file.with_suffix(".graphml")))
    else:
        LOG.error(f'Version type {VERSION.type} not supported')
        exit(1)


if __name__ == '__main__':
    args: Args = parse_args(Args)[0]
    if args.version is None:
        DATASET = DatasetSchema.load_schema(args.dataset)
        for version in DATASET.versions.keys():
            LOG.info(f'Exporting synthetic version {args.dataset}:{version}')
            args.version = version
            run(args)
    else:
        run(args)
