import numpy as np
from tree2 import *
from visualize_trees import *
from helper_functions import *
from copy import deepcopy
import pickle

def node_id():
    """
    generator function - is used by mark_nodes(tree_) to assign nodes to tree_
    """
    for node in range(1000):
        yield node


def mark_nodes(tree_):
    """
    uses node_id() generator to assign nodes to tree (depth first)
    """
    tree_.node = next(node_i)
    if len(tree_.kids) is not 0:
        for kid in tree_.kids:
            mark_nodes(tree_.kids[kid])


def pop_list(tree_):
    """
    populates a list with tree_ nodes
    the format of the resulting tree_list is
    [ #node (int), attribute that is split (int), kids (list), label (list), rate of positives (float) ]
    """
    kids = []
    if len(tree_.kids) is not 0:
        for i in tree_.kids:
            kids.append(tree_.kids[i].node)
    tree_list.append([tree_.node, tree_.op, kids, tree_.label, tree_.rate])
    if len(tree_.kids) is not 0:
        for i in tree_.kids:
            pop_list(tree_.kids[i])


if __name__ == "__main__":

    # LOAD DATA
    data = loadmat('Data/cleandata_students.mat')
    # data = loadmat('Data/noisydata_students.mat')

    x = data['x']
    y = data['y']
    N = x.shape[0]
    attributes = np.array(range(0, int(x.shape[1])))


    # RUN k-FOLD CROSS-VALIDATION
    K = 10
    errors_pruned = []
    errors_unpruned = []
    conf_mat_pruned = []
    conf_mat_unpruned = []

    all_unpruned_trees = []
    all_pruned_trees = []
    for fold_id in range(1, K+1):

        print('Fold %d of %d' % (fold_id, K))

        # SPLIT DATA PER FOLD
        xtrain, ytrain, xtest, ytest = make_k_fold_set(x, y, K, i=fold_id)

        unpruned_trees = {}
        pruned_trees = {}

        for emotion in range(1, 7):

            print(emotion)

            # TRAIN TREES
            binary_targets = target_binary_vector(ytrain, emotion)
            new_tree = decision_tree_learning(xtrain, attributes, binary_targets)

            # PREDICT BINARIES BASED ON TRAINED TREE
            ypred = new_tree.predict(xtest)

            # CALCULATE PERFORMANCE STATISTICS
            test_binary_targets = target_binary_vector(ytest, emotion)
            recall, precision = recall_precision(test_binary_targets, ypred)
            f1_score = f_score(recall, precision)

            error_ = error(test_binary_targets, new_tree.predict(xtest))

            unpruned_trees[emotion] = {'tree': new_tree, 'recall': recall,
                                      'precision': precision, 'f1_score': f1_score, 'error': error_}

            # COPY TRAINED TREE OBJECT
            trees = deepcopy(unpruned_trees)

            tree_ = trees[emotion]['tree']
            error_ = trees[emotion]['error']

            # MAKE TREE TO LIST FORMAT
            node_i = node_id()

            mark_nodes(tree_)

            tree_list = []

            pop_list(tree_)

            # START PRUNING
            # FOR A MAXIMUM OF ROUNDS EQUAL TO THE NUMBER OF NODES
            for i in range(len(tree_list)):

                error_pruned = {}
                # FOR EACH NODE IN THE TREE
                for node_ in range(1, len(tree_list)):

                    tree_list2 = deepcopy(tree_list)
                    
                    del tree_list2[node_]

                    ypred_list = test_tree_list(xtest, tree_list2)

                    error_pruned[node_] = error(test_binary_targets, ypred_list)

                # FIND BEST NODE TO PRUNE
                node_to_prune = min(error_pruned, key=error_pruned.get)
                error_pruned_ = error_pruned[node_to_prune]

                # CHECK THAT PRUNING IM,PROVES ERROR RATE
                if error_pruned_ < error_:

                    del tree_list[node_to_prune]

                    # print(node_to_prune)

                    error_ = error_pruned_

                # IF IT DOES NOT STOP PRUNING (greedy)
                else:
                    break

            # GET PERFORMANCE STATISTICS FOR FINAL PRUNED TREE
            recall, precision = recall_precision(test_binary_targets, test_tree_list(xtest, tree_list))
            f1_score = f_score(recall, precision)

            # ADD FINAL PRUNED TREE TO DICT OF PRUNED TREES
            pruned_trees[emotion] = {'tree_list': tree_list, 'error': error_,
                                     'precision': precision, 'recall': recall, 'f1_score': f1_score}

        # FINISHED PRUNING TREES
        # NOW WE HAVE A DICT OF 6 PRUNED TREES (one per emotion)
        # SAVE
        all_unpruned_trees.append(unpruned_trees)
        all_pruned_trees.append(pruned_trees)

        # CLASSIFY TEST SET WITH PRUNED AND UNPRUNED TREES AND GET SUMMARY STATISTICS
        y_unpruned = testTrees2(unpruned_trees, xtest)
        errors_unpruned.append(error(ytest, y_unpruned))
        conf_mat_unpruned.append(confusion_mat(ytest, y_unpruned))

        y_pruned = testTrees2(pruned_trees, xtest)
        errors_pruned.append(error(ytest, y_pruned))
        conf_mat_pruned.append(confusion_mat(ytest, y_pruned))
	
	
	
	#-----------Errors in 3 decimal places-------------#
    errors_unpruned_2dec = [round(float(i), 3) for i in errors_unpruned]
    errors_pruned_2dec = [round(float(i), 3) for i in errors_pruned]
    # SHOW MEAN ERROR AND CONF MATRIX FOR PRUNED AND UNPRUNED TREES
    print('---------------UNPRUNED TREES---------------------')
    print('\nERRORS PER EMOTION')
    print(errors_unpruned_2dec)
    print('\nAVERAGE ERROR')
    print(round(np.mean(errors_unpruned),3))
    print('\nCONFUSION MATRIX')
    print(average_confusion_matrix(conf_mat_unpruned))
    print('\n-----------------PRUNED TREES---------------------')
    print('\nERRORS PER EMOTION')
    print(errors_pruned_2dec)
    print('\nAVERAGE ERROR')
    print(round(np.mean(errors_pruned),3))
    print('\nCONFUSION MATRIX')
    print(average_confusion_matrix(conf_mat_pruned))


precisions = [np.mean([un_tree[em]['precision'] for un_tree in all_unpruned_trees]) for em in range(1, 7)]
recall = [np.mean([un_tree[em]['recall'] for un_tree in all_unpruned_trees]) for em in range(1, 7)]
