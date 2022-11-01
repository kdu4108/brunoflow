import numpy as np
import numpy.typing as npt
from typing import Any, List, Tuple, Union


def insert_into_list(existing_ranks: list, candidate_val: float, maxlen: int = 10) -> list:
    """Track the top-k (maxlen) values in a list"""
    existing_ranks.append(candidate_val)
    for i in range(len(existing_ranks) - 1, 0, -1):
        if existing_ranks[i] > existing_ranks[i - 1]:
            # Keep swapping until we hit a an element where the candidate is not greater
            existing_ranks[i], existing_ranks[i - 1] = existing_ranks[i - 1], existing_ranks[i]
        else:
            break

    if len(existing_ranks) > maxlen:
        existing_ranks.pop()

    return


def check_top_k_invariant(max_pos_grads: List[tuple], max_neg_grads: List[tuple]):
    for i in range(len(max_pos_grads) - 1):
        assert (max_pos_grads[i][0] >= max_pos_grads[i + 1][0]).all()

    for i in range(len(max_neg_grads) - 1):
        assert (max_neg_grads[i][0] <= max_neg_grads[i + 1][0]).all()

    assert (max_pos_grads[0][0] >= max_neg_grads[0][0]).all()
