import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import StratifiedKFold
from sklearn.svm import SVC
from sklearn import preprocessing

init_net 	 = np.array([1,1,1,2,2,2,2,2,1,2,2,2,2,1,2,2,2,2,2,2,2,1,1,2,2,2,2,1,2,2,2,1,2,2,2,2,2,1,1,1,2,2,1,2,2,2,1,2,2,2,1,2,1,1,1,2,2,2,2,2,2,1,2,2,2,1,2,1,1,1,2,1,2,1,1,2,1,1,2,2,1,1,2,2,1,1,1,2,2,1,2,2,1,2,2,2,2,1,2,2,1,1,2,2,1,1,2,1,1,1,1,1,2,2,2,2,2,1,1,1,1,1,2,1,1,2,2,2,2,2,1,2,1,1,1,2,1,2,2,2,2,2,1,1,2,1,1,2,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,1,1,2,1,2,1,1,2,2,1,2,2,2,2,2,2,1,1,1,2,2,1,1,2,2,2,2,2,1,2,1,1,1,2,1,1,2,2,1,1,2,1,2,1,1,2,1,2,1,2,1,2,2,2,2,2,1,1,2,1,2,1,2,1,1,1,1,2,1,2,2,2,2,2,1,1,1,1,2,2,2,1,2,1,1,1,1,2,2,1,1,2,2,1,2,1,2,1,2,2,2,1,1,1,1,1,2,2,2,2,1,1,1,1,1,2,1,1,1,1,2,2,2,2,2,1,2,1,2,1,1,2,2,2,2,2,1,2,2,2,2,2,1,1,2,2,2,2,2,2,1,2,2,1,2,2,1,2,2,2,2,2,1,1,2,1,2,1,2,1,1,2,1,2,2,2,1,2,1,1,2,2,2,2,2,1,1,2,2,2,1,1,2,2,2,1,1,1,2,1,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,2,2,2,2,2,2,1,1,1,1,2,2,1,1,1,2,2,2,2,2,2,2,1,1,1,2,1,2,2,2,2,2,1,2,2,2,1,2,2,2,1,2,2,2,2,2,2,1,2,2,1,1,1,2,2,2,1,1,2,1,1,1,2,2,2,2,2,1,2,1,1,2,2,2,1,2,2,2,1,1,1,1,1,1,2,1,2,1,1,1,1,1,2,2,2,2,2,2,2,2,1,2,1,2,1,1,2,2,1,2,2,2,2,2,1,2,2,2,1,2,1,2,2,2,2,2,1,2,2,1,1,2,1,2,2,2,2,2,1,2,1,2,1,1,2,2,1,2,2,2,2,2,1,1,2,1,1,1,1,2,2,1,2,2,1,1,1,1,1,2,2,2,1,1,2,1,2,2,2,1,2,1,2,1,1,2,2,1,2,2,1,1,1,1,1,2,2,2,2,2,2,1,1,1,2,2,2,1,2,2,1,1,2,2,2,1,1,2,2,2,1,1,1,1,1,1,1,1,2,2,2,2,2,2,1,2,2,2,1,1,2,2,2,2,2,2,2,1,2,1,2,2,1,1,2,2,2,2,2,2,2,1,1,2,1,1,2,2,2,2,2,1,2,1,1,1,1,2,2,2,2,2,2,2,2,2,2,1,1,1,2,2,2,1,2,2,1,2,1,2,2,1,2,2,1,2,2,1,1,1,1,1,1,1,1,2,2,2,2,2,2,1,2,2,2,2,2,2,1,2,2,2,1,2,2,1,2,2,2,2,2,1,2,2,1,1,2,2,2,2,1,2,2,1,1,2,2,1,1,1,2,1,2,2,2,1,2,2,1,1,2,2,2,2,1,1,2,1,2,1,1,2,1,2,1,2,2,2,2,2,2,1,1,1,2,1,2,2,2,2,1,2,2,1,1,2,2,1,1,2,2,1,2,2,2,1,2,2,1,1,2,2,2,2,2,2,2,2,1,1,1,1,1,1,2,2,2,2,2,2,2,1,1,2,2,2,2,2,1,2,1,1,1,2,2,2,2,1,2,1,1,2,2,2,1,1,2,1,2,2,1,2,2,2,1,1,1,2,1,2,2,2,2,2,2,2,2,1,2,1,2,1,2,1,2,1,2,2,2,2,2,2,1,2,1,2,1,1,2,2,2,2,2,2,1,2,1,1,1,1,2,1,2,2,2,1,1,1,2,2,2,2,2,2,1,1,2,1,1,1,2,1,2,2,2,2,1,1,1,2,2,2,2,1,2,2,2,1,1,1,1,1,2,2,1,1,2,2,2,2,2,2,1,1,2,1,1,2,2,2,2,2,2,2,1,1,1,2,2,1,2,2,1,2,2,1,2,1,1,1,1,1,2,2,2,2,2,2,1,2,1,2,2,2,1,2,1,2,2,1,2,1,1,1,1,2,1,2,2,2,2,1,2,1,1,2,1,2,2,1,1,2,1,2,1,1,2,2,2,2,2,2,2,2,2,2,1,2,2,1,1,2,1,1,1,2,2,2,1,1,1,2,2,1,2,2,2,1,1,2,2,1,1,2,1,2,2,1,1,2,1,2,2,1,2,2,2,2,2,1,2,2,2,2,2,1,1,1,1,2,2,1,2,1,1,1,2,1,2,2,1,1,2,2,2,1,1,2,1,1,1,1,2,1,2,2,2,2,2,2,2,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,1,2,2,1,2,1,2,2,2,1,1,1,1,2,2,1,2,2,2,1,1,2,1,2,2,2,2,2,2,2,1,1,2,1,1,1,2,1,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,1,2,1])
init_ill 	 = np.array([1,2,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,1,2,2,2,2,2,2,2,1,2,2,1,2,2,2,2,2,2,2,2,2,2,1,1,2,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,1,2,2,1,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,1,1,1,2,2,2,2,2,2,2,1,1,2,2,1,1,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,1,2,2,1,2,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,2,2,1,1,2,2,2,2,2,2,2,2,1,2,2,2,1,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,2,1,2,2,2,2,2,2,2,1,2,1,2,2,2,2,2,2,2,2,2,2,1,2,1,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,1,2,2,1,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,1,2,2,2,2,2,2,2,2,2,1,2,2,2,1,2,2,2,2,2,2,2,2,2,1,2,1,2,2,2,2,2,2,1,2,1,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,2,1,1,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,2,1,2,2,1,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,1,2,2,2,2,2,2,2,2,2,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,2,1,2,2,2,2,2,2,1,1,1,1,1,1,2,1,2,1,1,1,2,1,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,1,2,1,2,2,2,2,2,2,2,1,2,2,2,1,2,2,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,1,2,1,2,2,2,2,2,2,1,2,2,2,2,1,1,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,1,2,1,1,2,1,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,1,2,2,2,2,2,2,2,1,1,2,1,2,2,2,1,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,2,2,1,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,2,2,1,2,1,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,1,1,2,2,2,2,2,2,2,2,2,2,2,2,1,2,1,2,2,1,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,1,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,2,1,1,2,2,2,1,2,1,2,2,2,2,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,2,2,2,2,1,2,2,1,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,2,2,2,1,2,1,2,2,2,2,2,1,1,2,2,1,2,2,2,2,2,2,2,2,2,2,2,2,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,2,1,1,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,1,1,2,2,2,1,1,2,1,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2])
init_ideo 	 = np.array([1,1,2,2,2,2,2,2,1,1,2,2,2,1,2,2,2,2,2,2,2,1,1,2,2,2,2,2,1,2,2,1,2,1,2,2,2,1,2,1,2,2,1,2,2,1,1,1,2,2,1,2,1,1,1,2,2,2,2,1,2,2,2,2,2,1,2,1,2,2,1,1,1,1,1,2,1,1,2,2,1,2,2,2,1,1,1,2,1,1,2,2,1,2,2,1,2,2,1,2,1,1,2,1,1,1,2,1,2,1,1,1,2,2,2,2,2,1,2,1,1,1,1,1,1,2,2,2,2,2,1,2,1,1,2,2,2,2,2,2,2,2,1,1,2,2,1,2,2,2,2,1,1,1,2,2,2,2,2,1,2,2,2,1,1,2,2,2,1,2,1,2,1,2,2,2,2,1,2,1,2,2,2,2,1,1,2,2,2,2,2,2,1,2,1,1,2,1,1,2,2,2,1,2,1,2,2,1,2,2,2,1,2,1,2,2,1,2,2,1,1,2,1,2,1,2,2,1,1,1,1,2,2,2,1,1,2,1,1,1,1,2,2,2,2,1,1,2,2,1,2,2,2,1,2,1,1,1,1,2,1,2,2,1,1,1,1,1,1,2,2,2,1,2,1,1,1,2,2,1,1,1,1,2,2,1,2,2,1,1,1,2,1,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,1,2,2,1,2,2,2,2,2,2,2,2,1,2,1,1,1,1,2,1,2,1,2,1,2,1,1,2,2,2,2,2,1,1,1,2,2,2,1,2,2,2,1,1,1,2,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,2,1,1,1,2,2,2,1,2,1,1,2,1,2,2,1,1,1,2,2,1,2,2,2,2,1,2,1,1,2,2,2,2,2,2,1,2,2,2,1,2,2,2,1,2,2,2,1,2,1,1,2,2,1,2,1,1,2,1,1,2,2,1,2,1,2,2,2,1,1,1,1,1,1,2,2,2,1,2,2,2,1,1,2,1,1,1,2,1,2,1,1,2,1,1,2,2,2,2,2,2,2,2,1,2,2,2,2,1,1,2,2,2,2,1,2,2,1,2,2,2,1,2,1,2,2,2,2,1,1,2,2,1,2,2,1,2,1,1,2,2,1,2,1,2,1,2,2,2,1,2,2,2,1,2,1,1,2,1,1,1,2,2,2,1,1,1,1,1,1,1,1,2,2,2,1,2,1,2,2,2,1,1,2,1,2,1,1,2,1,1,2,2,1,1,1,1,1,2,2,2,2,2,2,1,2,1,1,2,2,1,2,2,2,2,2,2,2,1,1,2,2,2,1,2,2,1,2,1,1,1,1,2,2,1,2,2,1,1,2,2,2,2,2,2,2,2,1,1,2,2,1,1,2,1,1,1,2,2,2,1,2,1,2,2,1,1,2,1,2,2,2,2,2,1,1,1,1,1,1,2,2,2,2,2,1,2,2,2,2,1,1,1,2,2,2,1,2,2,1,2,1,2,2,1,1,2,2,2,2,1,1,2,1,1,1,1,1,2,2,1,1,1,1,1,2,2,2,2,2,2,1,2,2,2,1,1,2,1,2,2,2,2,2,1,2,2,2,1,2,2,2,2,1,2,2,1,2,2,2,1,1,1,2,1,2,2,2,1,2,2,1,1,1,2,2,1,1,2,2,1,2,1,1,2,1,2,2,2,2,1,2,1,2,1,2,1,2,2,1,2,2,2,1,2,2,2,1,2,2,1,2,1,2,2,2,2,2,1,1,2,2,1,2,2,1,1,2,2,2,2,1,1,1,1,2,1,2,2,2,2,1,1,2,1,1,2,2,1,1,2,2,2,1,1,2,2,2,2,1,2,1,1,1,2,2,2,2,1,2,1,2,2,1,2,2,2,2,1,1,2,1,2,2,2,2,2,2,2,2,2,2,2,2,1,2,1,2,1,2,2,2,2,2,1,1,2,1,2,1,1,2,2,1,1,2,1,1,2,1,2,1,1,2,1,2,1,2,1,1,2,1,2,1,2,2,1,1,2,2,2,2,2,2,1,2,2,2,2,1,2,2,1,2,2,1,1,2,2,2,1,2,1,1,1,2,1,1,2,2,2,1,2,2,2,1,1,2,2,1,2,2,2,2,2,2,1,1,1,1,2,2,1,2,2,1,2,2,1,2,2,1,2,2,1,2,2,2,2,2,2,1,2,2,2,1,2,1,1,1,1,2,1,1,1,1,2,1,2,2,2,2,2,2,2,1,1,1,1,1,1,2,1,1,2,1,2,1,2,2,2,2,2,2,2,2,1,1,2,1,2,2,1,1,2,1,1,1,2,2,2,2,1,1,2,2,2,2,2,2,1,1,2,2,1,1,2,2,2,2,1,1,2,2,2,2,1,1,2,2,2,2,1,2,1,2,2,1,1,2,2,1,2,2,1,2,1,1,2,2,2,2,2,2,2,1,2,2,1,1,1,1,1,1,1,1,1,2,2,2,2,1,1,1,1,1,2,1,1,2,2,2,1,2,2,2,2,2,2,1,2,2,1,2,2,2,1,2,1,2,1,1,2,2,1,2,2,2,1,1,2,2,2,2,2,2,2,2,2,1,1,2,1,1,1,2,1,2,2,2,2,2,1,2,2,2,1,2,2,2,2,2,1,2,2])
init_targets = np.array([1,1,2,2,2,1,2,2,2,2,1,1,1,1,1,2,2,2,2,2,2,2,1,1,1,1,1,1,1,2,2,2,2,2,2,2,1,1,1,1,1,1,1,2,2,2,2,2,2,2,1,1,2,1,1,1,1,2,2,2,2,2,1,1,1,1,2,1,2,2,2,2,2,1,1,1,1,1,2,2,2,2,2,2,1,1,1,1,1,1,2,2,2,2,2,2,1,1,1,2,2,2,2,2,2,1,1,1,1,1,1,1,2,2,2,2,2,2,1,1,1,1,1,1,1,2,2,2,2,1,2,2,2,1,1,1,1,2,2,2,2,2,2,2,1,1,1,1,1,2,2,2,2,2,2,2,1,1,1,1,2,2,2,2,2,2,2,1,1,1,1,1,1,1,2,2,2,2,2,2,1,1,1,1,2,1,2,2,2,2,2,1,1,1,1,1,1,2,1,2,2,2,2,2,1,1,1,1,1,1,2,2,2,2,2,2,1,1,1,1,1,2,1,2,2,2,2,2,1,1,1,2,2,2,2,2,2,1,1,1,1,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,1,1,1,1,2,2,2,2,2,2,2,1,1,1,1,1,1,1,2,2,2,2,2,2,2,1,1,1,2,2,2,2,2,2,1,1,1,1,1,1,1,1,2,2,2,2,2,2,1,1,1,2,1,1,1,2,2,1,2,2,2,1,1,1,2,2,2,2,2,2,1,1,1,1,1,2,2,1,2,2,2,2,1,1,1,1,2,2,2,2,2,2,1,1,1,1,1,1,1,2,2,2,2,2,2,2,1,1,1,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,1,1,1,1,2,1,1,2,2,2,2,2,2,1,1,1,1,1,1,1,2,2,2,2,2,2,1,1,1,1,2,2,2,2,2,2,1,1,1,1,1,1,1,2,2,2,2,2,2,1,1,1,1,1,2,1,2,2,2,2,2,1,1,2,2,1,2,2,2,2,1,1,1,1,1,2,2,2,2,2,2,1,1,1,1,2,1,2,2,2,2,2,2,1,1,1,1,1,1,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,1,1,1,1,2,2,2,2,2,2,2,1,1,1,1,1,1,1,2,2,2,2,2,2,1,1,1,1,1,1,2,2,2,2,2,2,1,1,1,2,1,2,2,2,2,2,1,1,1,1,1,1,2,2,2,2,2,2,1,1,1,1,1,1,1,1,2,2,2,2,2,2,1,1,1,1,2,2,2,2,2,2,1,1,2,2,2,2,2,2,2,1,1,1,1,1,1,2,1,1,2,2,2,2,2,2,1,1,1,1,1,1,1,2,2,2,2,2,2,2,1,1,1,1,1,2,2,2,2,2,2,2,1,1,1,1,1,2,2,2,2,2,2,1,1,1,1,1,1,1,2,2,2,2,2,2,1,1,1,1,1,1,2,2,2,2,2,2,1,1,1,1,1,1,2,2,2,2,2,2,1,1,1,1,2,2,2,2,2,2,1,1,1,1,2,2,2,2,2,2,1,1,1,1,1,1,2,1,1,2,2,2,2,2,2,1,1,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,1,1,1,1,1,1,2,2,2,2,2,2,2,1,1,1,2,2,2,2,2,2,1,1,1,1,1,1,2,2,2,2,2,2,1,1,1,1,1,1,1,2,1,2,2,2,2,2,1,1,1,1,1,2,2,2,2,2,2,1,1,1,1,1,1,2,2,2,2,2,2,1,1,1,2,2,2,2,2,2,1,1,1,1,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,2,2,2,2,2,1,2,2,1,1,1,1,2,2,2,2,2,2,2,1,1,1,1,1,1,2,1,1,2,2,1,2,2,2,2,1,1,1,1,1,2,2,2,2,2,2,1,1,1,1,1,1,2,2,2,2,2,2,1,1,1,2,2,2,2,2,2,1,1,1,1,1,2,1,2,2,2,2,2,1,1,1,1,2,2,2,2,2,2,1,1,1,1,1,1,1,1,2,2,2,2,2,2,1,1,1,1,1,1,2,2,2,2,2,2,2,1,1,1,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,1,1,1,1,2,1,2,2,2,2,2,2,1,1,1,1,1,1,2,2,2,2,2,2,1,1,1,1,1,2,2,1,2,2,2,2,1,1,1,1,1,2,2,2,2,2,2,1,1,1,1,2,1,2,2,2,2,2,1,1,1,1,2,2,2,2,2,2,1,1,1,1,1,1,2,2,2,1,2,2,2,1,1,1,1,1,2,2,2,2,2,2,2,1,1,1,1,1,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,1,1,1,1,2,2,2,2,2,2,2,1,1,1,1,1,2,1,2,2,2,2,2,1,1,1,2,1,1,1,2,2,2,2,2,1,1,1,1,2,1,2,2,2,2,2,1,1,1,1,2,2,2,2,2,2,1,1,1,1,1,1,1,1,2,2,2,2,2,2,1,1,2,1,2,2,2,2,2,1,1,1])

net = []
ill = []
ideo = []
targets = []

def check_prev(inet, iill, iideo, itargets):
	for i in range(len(targets)):
		if net[i] == inet and ill[i] == iill and ideo[i] == iideo and targets[i] == itargets:
			return True
	return False	

def create_dataset_and_targets():
	for i in range(len(init_targets)):

		if not (init_net[i] == init_ill[i] and init_ill[i] == init_ideo[i] and init_net[i] != init_targets[i]):
			if not check_prev(init_net[i], init_ill[i], init_ideo[i], init_targets[i]):
				net.append(init_net[i])
				ill.append(init_ill[i])
				ideo.append(init_ideo[i])
				targets.append(init_net[i])

	return np.array(net), np.array(ill), np.array(ideo), np.array(targets)

def combine_and_process_dataset():
	net, ill, ideo, targets = create_dataset_and_targets()

	data = []
	for i in range(len(targets)):
		data.append([float(net[i]), float(ill[i]), float(ideo[i])])

	dataset = preprocessing.scale(data)

	return dataset, targets

if __name__ == "__main__":
	dataset, targets = combine_and_process_dataset()

	C_range = np.arange(0.1, 16, 0.2)
	gamma_range = np.arange(0.1, 16, 0.2)
	param_grid = dict(gamma=gamma_range, C=C_range)
	# cv = StratifiedShuffleSplit(targets, random_state=42)
	cv = StratifiedKFold(targets)
	grid = GridSearchCV(SVC(class_weight='auto'), param_grid=param_grid, cv=cv, scoring='accuracy')
	grid.fit(dataset, targets)
	print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))
