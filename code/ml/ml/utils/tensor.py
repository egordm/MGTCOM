import torch


def partition_values(vs, ranges):
    partitions = []
    for i, (start, end) in enumerate(ranges):
        size = int(torch.sum((vs >= start) & (vs < end)))
        partitions.append(size)

    partitions = torch.sort(vs).values.split(partitions)

    return partitions
