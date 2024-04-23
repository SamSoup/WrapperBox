from typing import List
from datasets import DatasetDict


def partition_indices(N, M) -> List[int]:
    """
    Divides a sequence of indices from 1 to N into M partitions

    Parameters:
    - total_elements (int): The total number of elements to be divided.
    - num_sections (int): The number of sections to divide the elements into.

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
