# Example-based Explanation for Light Gradient-Boosting Machine
# https://en.wikipedia.org/wiki/LightGBM
# It is similar to DecisionTreeClassifier, but actually fits one tree per
# class, and it is much faster than sklearn's implementation
from lightgbm import LGBMModel
from tqdm import tqdm
from utils.models.dt import get_leaf_ids, get_predictions
from utils.partition import get_partition_X
from ExampleBasedExplanations.interface import ExampleBasedExplanation
from scipy.spatial.distance import cdist
from typing import List, Union
import numpy as np


class LGBMExampleBasedExplanation(ExampleBasedExplanation):
    def __init__(self) -> None:
        super().__init__()

    # def _get_explanation_indices_leaf_batched(
    #     self,
    #     M: int,
    #     clf: LGBMModel,
    #     train_embeddings: np.ndarray,
    #     test_embeddings: np.ndarray,
    # ) -> List[List[int]]:
    #     """
    #     Return the closest leaf examples for decision tree in a smart way by
    #     batching test examples per leaf, and computing distances in a vectorized
    #     fashion for each leaf node for parallelism.

    #     Args:
    #         clf (Union[LGBMModel, DecisionTreeClassifier]):
    #         M (int): number of examples to return for explanation, = K here
    #         dataset (DatasetDict): must have at least {
    #             "train": ['text', 'label'...]
    #             "test": ['text', 'label'...]
    #         }
    #         train_embeddings (np.ndarray): (num_train_examples, hidden_size)
    #         test_embeddings (np.ndarray): (num_test_examples, hidden_size)

    #     Returns:
    #         np.ndarray: The M closest leaf example indices for each test example,
    #             (num_test_examples, M)
    #     """
    #     leaf_ids_train = get_leaf_ids(clf, train_embeddings)
    #     leaf_ids_test = get_leaf_ids(clf, test_embeddings)

    #     unique_leaf_ids_test = set(leaf_ids_test)
    #     ex_indices = [[] for _ in range(test_embeddings.shape[0])]
    #     for leaf_id in tqdm(
    #         unique_leaf_ids_test, "Getting Expl Indices for DT Batched"
    #     ):
    #         # Get all test examples in the leaf id
    #         _, test_indices = get_partition_X(leaf_ids_test, leaf_id)

    #         # Get all train examples in the leaf id
    #         _, train_indices = get_partition_X(leaf_ids_train, leaf_id)

    #         # Compute distance from test examples to train examples
    #         # shape(test_leaf_subset.shape[0], train_leaf_subset.shape[0])
    #         dist_mat = cdist(
    #             test_embeddings[test_indices],
    #             train_embeddings[train_indices],
    #             metric="euclidean",
    #         )

    #         # Order the distances by indices
    #         if M is None:
    #             # assume we want all train exampes
    #             M = test_indices.shape[0]

    #         # NOTE: one potential complications here is that there may not be M
    #         # examples in a leaf, if that leaf is particularly small
    #         top_k = np.argsort(dist_mat, axis=1)[:, :M]

    #         # This indexing might be confusing: we are essentially ordering the
    #         # training indices according to their sorted rank in top_k
    #         # shape: (test.shape[])
    #         sorted_expl_indices = train_indices[top_k]

    #         # Set the appropriate train example (expl) indices for each
    #         # test example
    #         for test_idx, expl_ind in zip(test_indices, sorted_expl_indices):
    #             ex_indices[test_idx] = expl_ind
    #     return ex_indices

    def _get_explanation_indices_brute_force(
        self,
        M: int,
        clf: LGBMModel,
        train_embeddings: np.ndarray,
        test_embeddings: np.ndarray,
    ) -> List[List[int]]:
        """
        Return the closest leaf examples for the Light Gradient Boosting Machine
        using manual, brute force search for each test example.

        This method is suitable for when number of test examples ~= number of
        leaf nodes.

        Args:
            clf (LGBMModel):
            M (int): number of examples to return for explanation, = K here
            dataset (DatasetDict): must have at least {
                "train": ['text', 'label'...]
                "test": ['text', 'label'...]
            }
            train_embeddings (np.ndarray): (num_train_examples, hidden_size)
            test_embeddings (np.ndarray): (num_test_examples, hidden_size)

        Returns:
            np.ndarray: The M closest leaf example indices for each test example,
                (num_test_examples, M)
        """
        # Assuming that there is only one tree, then for a multiclass problem,
        # it will fit one tree per class (otherwise m tree per n class)
        # In the one tree scenario, since only one tree with the highest
        # logit is the one used for inference, we actually only care about
        # the tree used for inference and its leaf id for this sample
        predictions = get_predictions(clf, test_embeddings)
        # (num_test_exs, n_trees * n_classes)
        leaf_ids_test_per_tree = get_leaf_ids(clf, test_embeddings)
        leaf_ids_test = leaf_ids_test_per_tree[
            np.arange(leaf_ids_test_per_tree.shape[0]), predictions
        ]
        # for each test example, to get its corresponding leaf examples,
        # we need to first identify the tree actually used in infernece
        # (using predictions), then find all examples in that inference tree
        # with the same leaf number as the test example
        # This process will be cached to speed up retrieval
        # (num_train_exs, n_trees * n_classes)
        leaf_ids_train_per_tree = get_leaf_ids(clf, train_embeddings)

        indices = []
        for x_test, pred, leaf_idx in tqdm(
            zip(test_embeddings, predictions, leaf_ids_test),
            desc="Getting Expl Indices for DT Manually",
            total=predictions.size,
        ):
            # [0, num_train_examples)
            leaf_ids_train = leaf_ids_train_per_tree[:, pred]
            train_indices = np.arange(leaf_ids_train.shape[0])
            # only obtain the training examples in the same leaf
            # as the test example
            train_leaf_neighbor_mask = leaf_ids_train == leaf_idx
            train_indices_subset = train_indices[train_leaf_neighbor_mask]
            # unforunately, the set of examples to compute the distance is
            # different per input examples, and thus is not easily paralleled
            # TODO: could simply group test examples by leaf, then compute
            # all in parallel to save some time?
            dist_mat = cdist(
                x_test.reshape(1, -1),
                train_embeddings[train_leaf_neighbor_mask],
            )
            # one potential complications here is that there may not be M
            # examples in a leaf, if that leaf is particularly small
            top_k = np.argsort(dist_mat, axis=1)[:, :M].flatten()
            indices.append(train_indices_subset[top_k])

        return indices

    def get_explanation_indices(
        self,
        M: int,
        clf: LGBMModel,
        train_embeddings: np.ndarray,
        test_embeddings: np.ndarray,
    ) -> List[List[int]]:
        return self._get_explanation_indices_brute_force(
            M=M,
            clf=clf,
            train_embeddings=train_embeddings,
            test_embeddings=test_embeddings,
        )
