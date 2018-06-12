import pickle
from helper_functions import *

#Loads pickle with stored trees
trees_ = pickle.load( open( "save.p", "rb" ) )

#Predictions
y_pruned = testTrees2(trees_, xtest)
error(ytest, y_pruned)