from scipy.io import loadmat
from scipy.stats import mode
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd


class tree():

    def __init__(self, op = '', kids = {}, label = [], rate = None, node=None):
        self.op = op
        self.kids = kids
        self.label = label              
        self.rate = rate  # rate of positives in node
        self.node = node

    def add_child(self, child, index):
        if len(self.kids) < 2: 
            self.kids[index]=child
        else:
            print("Already 2 leaf nodes. Can't add anymore")
    
    def predict_single(self, example):
        """
        Use trained tree to predict class for new example
        Works with single or multiple examples
        """
        current_tree = self
        while len(current_tree.label) == 0:
            op = int(current_tree.op)
            current_tree = current_tree.kids[example[op]]
            if example[op] not in [0, 1]:
                print('Error')
                print(current_tree.label)
        return current_tree.label

    def predict(self, examples):
        """
        Use trained tree to predict class for new example
        Works with single or multiple examples
        """
        if len(examples.shape) == 1: # single example
            predicted = self.predict_single(examples)
        else:
            predicted = [self.predict_single(example) for example in examples]
        return np.array(predicted)
   

def choose_best_attribute(examples, attributes, binary_targets):
    """
    Choose best attribute to split data on based on Information Gain
    """
    gain = []
    # total positives
    p = binary_targets.sum().astype(float)
    # total negatives
    n = binary_targets.shape[0] - p
    for i in attributes:
        # if examples have unique value for attribute i there is no information gain
        if len(np.unique(examples[:, i])) == 1:
            gain.append(0)
        else:
            idx0 = examples[:, i] == 0
            p0 = (binary_targets[idx0] == 1).sum()
            n0 = (binary_targets[idx0] == 0).sum()
            idx1 = examples[:, i] == 1
            p1 = (binary_targets[idx1] == 1).sum()
            n1 = (binary_targets[idx1] == 0).sum()
            remainder = ((p0+n0)/(p+n))*I(p0, n0) + ((p1+n1)/(p+n))*I(p1, n1)
            gain.append(I(p, n) - remainder)
    return attributes[gain.index(max(gain))]


def I(p, n):
    """
    Information for ratios of positive and negative values p,n (Entropy)
    """
    p = float(p)
    n = float(n)
    # special case - I = 0 when sample contains unique values
    if (p == 0.0) or (n == 0.0):
        return 0
    else:
        return - (p/(p+n)) * math.log(p/(p+n), 2) - (n/(p+n)) * math.log(n/(p+n), 2)


def rate_of_positives(binary_targets):
    counts = [(binary_targets == 0).sum(), (binary_targets == 1).sum()]
    if counts[0] == 0:
        return 1.
    elif counts[1] == 0:
        return 0.
    else:
        return counts[1]/sum(counts)


def majority_value(binary_targets):
    vals, counts = np.unique(binary_targets, return_counts=True)
    ind = np.argmax(counts)
    mode = vals[ind]
    return int(mode)


def decision_tree_learning(examples, attributes, binary_targets):
    """
    Recursive Decision Tree
    Each attribute is used only once
    Assumes binary values for all attributes - [0, 1] hardcoded
    """
    rate = rate_of_positives(binary_targets)
    if len(np.unique(binary_targets)) == 1:
        # All elements are the same, take the first element
        return tree(op='All examples are the same', label=binary_targets[0],
                    rate=rate)
    elif len(attributes) == 0:
        # No more attributes to use, take the majority vote
        mode = majority_value(binary_targets)
        return tree(op='Empty attributes', label=[mode],
                    rate=rate)
    else:
        best_attribute = choose_best_attribute(examples, attributes, binary_targets)
        # grow tree with best_attribute as root
        theTree = tree(op=str(best_attribute), kids={}, rate=rate)
        # remove used attribute
        attributes = np.delete(attributes, np.where(attributes == best_attribute))
        for value in [0, 1]:
            # find indices for values of best_attribute
            value_idx = examples[:, best_attribute] == value  # indices
            # split examples and targets accordingly
            examples_i = examples[value_idx, :]
            binary_targets_i = binary_targets[value_idx]
            # if there are no examples with that value stop that branch here
            if examples_i.shape[0] == 0:
                return tree(label=[majority_value(binary_targets)], rate=rate)
            # otherwise continue growing the tree
            else:
                subtree = decision_tree_learning(examples_i, attributes, binary_targets_i)
                theTree.add_child(subtree, value)
        return theTree


def decision_tree_learning2(examples, attributes, binary_targets, depth=0, max_depth=None):
    """
    Recursive Decision Tree with Max Depth
    Each attribute is used only once
    Assumes binary values for all attributes - [0, 1] hardcoded
    """
    if (max_depth is not None) and (depth == max_depth):
        # Maximum depth is reached, we stop here
        mode = majority_value(binary_targets)
        return tree(op='Maximum depth is reached',
                    label=[mode])
    elif len(np.unique(binary_targets)) == 1:
        # All elements are the same, take the first element
        return tree(op='All examples are the same',
                    label=binary_targets[0])
    elif len(attributes) == 0:
        # No more attributes to use, take the majority vote
        mode = majority_value(binary_targets)
        return tree(op='Empty attributes',
                    label=[mode])
    else:
        best_attribute = choose_best_attribute(examples, attributes, binary_targets)
        # grow tree with best_attribute as root
        theTree = tree(op=str(best_attribute),
                       kids={})
        # remove used attribute
        attributes = np.delete(attributes, np.where(attributes == best_attribute))
        for value in [0, 1]:
            # find indices for values of best_attribute
            value_idx = examples[:, best_attribute] == value  # indices
            # split examples and targets accordingly
            examples_i = examples[value_idx, :]
            binary_targets_i = binary_targets[value_idx]
            # if there are no examples with that value stop that branch here
            if examples_i.shape[0] == 0:
                return tree(label=[majority_value(binary_targets)])
            # otherwise continue growing the tree
            else:
                subtree = decision_tree_learning2(examples=examples_i,
                                                 attributes=attributes,
                                                 binary_targets=binary_targets_i,
                                                 depth=depth+1,
                                                 max_depth=max_depth)
                theTree.add_child(subtree, value)
        return theTree


def error(actual_targets, predicted_targets):
    """
    Error rate for N-class classification
    """
    actual_targets = actual_targets.ravel()
    predicted_targets = predicted_targets.ravel()
    return 1. - (predicted_targets == actual_targets).sum() / float(actual_targets.shape[0])


def count(actual_targets, predicted_targets, i, j):
    """
    count occurences where actual_target is i and predicted_target is j
    """
    idx_i = actual_targets == i
    return (predicted_targets[idx_i] == j).sum()


def confusion_mat(actual_targets, predicted_targets, norm=False):
    """
    Confusion Matrix for N-class classification
    If norm is True a normalized (per actual class) matrix is returned
    """
    assert len(predicted_targets) == len(actual_targets)
    labels = np.unique(np.concatenate((predicted_targets, actual_targets))) # all unique labels
    conf_mat = np.array(
        [[count(actual_targets, predicted_targets, i, j) for j in labels] for i in labels])
    if norm:
        conf_mat = (conf_mat.T / conf_mat.sum(axis=1)).T
    return conf_mat


def average_confusion_matrix(conf_mats):
    """
    calculates teh average given a list of confusion matrices
    """
    N = conf_mats[0].shape[0]
    av_conf_mat = [[np.mean([conf_[k][j] for conf_ in conf_mats]) for j in range(N)] for k in range(N)]
    return np.array(av_conf_mat)


def recall_precision(actual_targets, predicted_targets, percent=False):
    """
    For binary classification
    """
    assert len(predicted_targets) == len(actual_targets)
    # indices fro positives and negatives
    neg_idx = predicted_targets == 0
    pos_idx = predicted_targets == 1
    # TP, FP, FN, TN
    true_pos = float((actual_targets[pos_idx] == 1).sum())
    false_pos = float((actual_targets[pos_idx] == 0).sum())
    false_neg = float((actual_targets[neg_idx] == 1).sum())
    # true_neg = (actual_targets[neg_idx] == 0).sum()
    recall = true_pos / (true_pos + false_neg)
    precision = true_pos / (true_pos + false_pos)
    if percent:
        recall *= 100
        precision *= 100
    return recall, precision


def f_score(recall, precision, a=1):
    """
    Fa_score from precision and recall
    """
    return (1 + a) * (precision * recall) / (a * precision + recall)



