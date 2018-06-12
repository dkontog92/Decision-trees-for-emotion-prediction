import numpy as np


def target_binary_vector(y, idx):
    """
    returns a binary vector from y mapping:
        - idx values to 1
        - all other values to 0
    y   : N-class target vector
    idx : index of target value
    """
    # y==idx returns boolean array. Multiplying by 1 makes it binary
    return np.array(1 * (y == idx))


def make_k_fold_set(x, y, K, i=1):
    """
    split x, y in k equal parts
    return the ith part as test set and the rest as train set
    """
    N = x.shape[0]
    idx_test = np.zeros(N, dtype=bool)
    idx_test[np.arange((i-1)*(N//K), i*(N//K))] = True
    return x[~idx_test], y[~idx_test], x[idx_test], y[idx_test]


def testTrees1(T, x2):
    """
    The following assumes T has integer keys of the form 1,2,...
    if more than a single class is predicted, we choose one at random
    """
    pred = [T[i]['tree'].predict(x2) for i in range(1, len(T) + 1)]
    if len(x2.shape) == 1:
        N = 1
        # return array of length equal to the number of trees with 1s indicating positive class
        pred = np.concatenate(pred, axis=0) # binary 1d array
        pred = np.where(pred == np.max(pred))[0]
        # if more than one indices are predicted, choose at random
        if len(pred) > 1:
            pred = np.random.choice(pred) # select at random one of the 1 indices
    else:
        N = x2.shape[0]
        pred = np.concatenate(pred, axis=1) # binary 2d array
        # select at random one of the predicted indices for each row
        pred = np.array([np.random.choice(np.where(pred_ == np.max(pred_))[0]) for pred_ in pred])
    return pred.reshape(N, 1) + 1  # emotions numbering starts from 1


def testTrees2(T, x2):
    """
    The following assumes T has integer keys of the form 1,2,...
    if more than a single class is predicted, we choose the most confident one (greatest precision)
    if no class is prediuct6ed we choose the least confident one (smallest recall)
    """
    if 'tree' in T[1]:
        # tree input is tree class
        pred = [T[i]['tree'].predict(x2) for i in range(1, len(T) + 1)]
    elif 'tree_list' in T[1]:
        # tree input is tree list
        pred = [test_tree_list(x2, T[i]['tree_list']) for i in range(1, len(T) + 1)]
    else:
        print('Not recognised tree type. tree must be either tree.class or tree_list')

    def testTrees_single(pred_):
        pred_ = np.where(pred_ == 1)[0] # 1 is hardcoded now because we are looking exclusively for positives
        # if more than one indices are predicted, choose at random
        if len(pred_) > 1:
            # more than 1 positives, choose the most confident tree
            idx = np.argmax([T[i]['precision'] for i in pred_+1])
            pred_ = pred_[idx]
        elif len(pred_) == 0:
            # no positive, choose the tree with the smallest recall
            pred_ = np.argmin([T[i]['recall'] for i in range(1, len(T) + 1)])
        return pred_

    if len(x2.shape) == 1:
        N = 1
        pred = np.concatenate(pred, axis=0)  # binary 1d array
        pred = testTrees_single(pred)
    else:
        N = x2.shape[0]
        pred = np.concatenate(pred, axis=1)  # binary 2d array
        # select at random one of the predicted indices for each row
        pred = np.array([testTrees_single(pred_) for pred_ in pred])
    return (pred + 1).reshape(N, 1)  # emotions numbering starts from 1


def test_tree_list(examples, tree_list):

    def test_tree_list_single(example, tree_list):
        # find root - always root node id is 0
        current_node = find_item(tree_list, 0, 0)
        while not current_node[3]:  # while there is no label
            if current_node[2]:  # if it has kids
                child = current_node[2][example[int(current_node[1])]]
                if find_item(tree_list, child, 0):
                    # child node has been pruned
                    current_node = find_item(tree_list, child, 0)
                else:
                    # print('cant find node %d' % child)
                    return rate_to_choice(current_node[4])
            else:
                return rate_to_choice(current_node[4])
        return float(current_node[3][0])

    if len(examples.shape) == 1:
        return test_tree_list_single(examples, tree_list)
    else:
        N = examples.shape[0]
        res = [test_tree_list_single(example, tree_list)
               for example in examples]
        return np.array(res).reshape(N, 1)


def find_item(tree_list, value, idx):
    # only need to find a single item
    res = [item for item in tree_list if item[idx] == value]
    if res:
        res = res[0]
    else:
        res = None
    return res

def rate_to_choice(rate):
    return int(rate > 0.5)

