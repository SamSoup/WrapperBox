from typing import Iterable, List, Union, Tuple
from datasets import DatasetDict
import numpy as np


def get_partition_X(
    arr: np.ndarray, X: int
) -> Tuple[Iterable[int], Iterable[int], Iterable[int]]:
    """
    Given an array of items in `arr` of integer values (cluster ids, leaf ids),
    and a desired partition `X`, return the indices and subset of items in
    partition (cluster or leaf) X

    Args:
        arr (np.ndarray): 1D iterable of items to filter
        X (int): Desired partition

    Returns:
        Tuple[Iterable[int], Iterable[int], Iterable[int]]: Mask, indices of
            desired items in partition, as well as the partition subset.
            If
    """
    indices = np.arange(arr.shape[0])
    # only obtain the training examples in the same leaf
    # as the test example
    mask = arr == X
    indices_subset = indices[mask]
    subset = arr[mask]

    return mask, indices_subset, subset


def partition_indices(N: int, M: int) -> List[int]:
    """
    Divides a sequence of indices from 1 to N into M partitions

    Parameters:
    - N (int): The total number of elements to be divided.
    - M (int): The number of sections to divide the elements into.

    Returns:
        List[int]: - a list of the index of the partitions, namely:
        0:L[0], L[0]+1:L[1]... L[len-1]+1:N
    """
    section_size = N // M
    sections_indices = [min((i + 1) * section_size, N) for i in range(M - 1)]
    # manually populate the last section to back-load all
    # remainder examples when N does not divide M evenly
    sections_indices.append(N)

    return sections_indices


def filter_first_N_entries(dataset_dict: DatasetDict, N: int):
    """
    Filter down each split in a DatasetDict to only the first N entries.
    Useful when needing to test functionality of a small amonnt of code

    Args:
        dataset_dict (DatasetDict): The DatasetDict containing splits of a dataset.

    Returns:
        DatasetDict: The filtered DatasetDict.
    """
    filtered_dataset = DatasetDict(
        {
            split: dataset_dict[split].select(range(N))
            for split in dataset_dict.keys()
        }
    )

    return filtered_dataset
