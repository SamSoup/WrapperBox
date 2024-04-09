"""
Assuming a binary decision tree, we can view it as hierarchical clustering: 
for each training point, there is some path from root to leaf.  
At the root, the entire training data is bifurcated into two groups, and 
at each successive node, the training instances there are further bifurcated.

So an example-based way to think explanation via decision tree is playing a 
relative comparison version of "20-questions", where each node is question.  
If we think of the node as bifurcating two groups, then the question is whether 
the input example is more like the left group or the right group.  This means 
we need one or more representative examples of each group to show to the user. 
For simplicity, let's say one example.  We might compute the centroid of each 
group, then select whichever example is closest to it via some distance function 
(e.g., Euclidean, cosine, whatever). We can then explain the model prediction 
as a path of root-to-leaf relative comparisons.

This is similar to where Anubrata finds himself with ProtoTEx; just as the 
latent prototypes (ie cluster centroids) he uses cannot be directly shown 
to users, neither can the group centroids you find above.  Consequently, 
he picks one or more closest training examples to each prototype to show. 
In his case, he faces a tradeoff between the brevity vs. fidelity of the 
explanation, which acts like a summary of how the model behaves: showing 
only one training example is most concise but least accurate/faithful 
summary in approximating actual model behavior; showing all training 
examples would be 100% faithful but represent information overload for 
the user (effectively, not summarizing at all but showing the full working 
of the model).  A key point is that this tradeoff can be quantified: 
for k prototype neighbors shown (brevity), how accurately does prediction 
based on this truncated neighborhood approximate true model behavior (fidelity). 
Because ProtoTEx actually makes decisions this way, the explanations can 
be 100% faithful if you're willing to sacrifice brevity.

With the decision tree, however, the actual decisions are made based on 
feature-space splits, and the relative comparison method described earlier 
acts like a proxy model for the actual model behavior. That said, it seems 
like an empirical question how accurately this proxy model could 
simulate the original model, and I'm guessing it would be pretty accurate 
as we increase K (the number of examples in the group used to represent it. 
However, as you go deeper down the tree, the group sizes get progressively 
more sparse, with fewer and fewer examples in each group available. 
My guess is this means the approximation of which way to branch can be quite
accurate near the root but increasingly noisy as you approach the sparse leaves.
"""

import numpy as np
import random
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from utils import find_majority
from scipy.spatial.distance import cdist

class DecisionTreeProxyClassifier(BaseEstimator, ClassifierMixin):
    """
    This approach takes the tree as is, figure out which examples are 
    in the constructed tree nodes, and then infer based on the centroids
    of each tree node in one of two ways:
    
    1. Like 20 questions: use the fake "centroid" of the tree nodes and compare
    to inference example
    
    2. Take KNN from each tree node, and then go down the tree
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def _tree_to_cluster(self):
        """
        Figure out which
        examples belong to which node at each split of the
        tree, in a preorder traversal fashion 
        """
        tree_ = self.tree.tree_
        exs_indices = {}
        def preorder_traversal(node, indices):
            exs_indices[node] = indices
            subset = self.X_[indices]
            feat_idx = tree_.feature[node]
            if feat_idx != _tree.TREE_UNDEFINED:
                threshold = tree_.threshold[node]
                rule_fct = lambda ex: ex[feat_idx] <= threshold
                mask = np.apply_along_axis(rule_fct, 1, subset)
                left_indices, right_indices = indices[mask], indices[~mask]
                preorder_traversal(tree_.children_left[node], left_indices)
                preorder_traversal(tree_.children_right[node], right_indices)
            # else at leaf node, already done everything we need to
        preorder_traversal(0, np.arange(self.X_.shape[0]))
        return exs_indices

    def fit(self, X, y):
        ## Sanity Checks
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        
        self.tree = DecisionTreeClassifier(**self.kwargs).fit(self.X_, self.y_)

        # compute which examples are in which tree node
        self.cluster_indices = self._tree_to_cluster()

        # Return the classifier
        return self

    def _compute_dist_to_centroid(self, ex, cluster):
        centroid = cluster.mean(axis=0)
        dist = cdist(ex.reshape(1, -1), centroid.reshape(1, -1))
        return dist

    def _cluster_predict(self, ex):
        """
        Using the original decision tree node locations,
        predict using cluster centroids instead
        """
        tree_ = self.tree.tree_
        def preorder_traversal(node):
            feat_idx = tree_.feature[node]
            if feat_idx != _tree.TREE_UNDEFINED:
                l, r = tree_.children_left[node], tree_.children_right[node]
                l_dist = self._compute_dist_to_centroid(
                    ex, self.X_[self.cluster_indices[l]]
                )
                r_dist = self._compute_dist_to_centroid(
                    ex, self.X_[self.cluster_indices[r]]
                )
                # move left or right, accordingly
                if l_dist < r_dist:
                    return preorder_traversal(l)
                elif l_dist > r_dist:
                    return preorder_traversal(r)
                else:
                    # break ties randomly
                    c = random.randint(0, 1)
                    return preorder_traversal([l, r][c])
            else:
                indices = self.cluster_indices[node]
                labels = self.y_[indices]
                p = find_majority(labels)
                # vote using the labels
                return p
        return preorder_traversal(0)
    
    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X)
        return np.apply_along_axis(
            func1d=self._cluster_predict, axis=1, arr=X
        )

    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        if deep:
            return self.tree.get_params()
        else:
            return self.__dict__

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
