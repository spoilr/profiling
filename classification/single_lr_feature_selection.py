"""
Logistic Regression Classification
Single LR
Feature selection is applied before
"""

print(__doc__)

import sys
sys.path.insert(0, 'utils/')
sys.path.insert(0, 'feature context/')
from load_data import *
from project_data import *
from parse_theme import *
from feature_selection_before import *
from parameters import CV_PERCENTAGE_OCCURENCE_THRESHOLD
from cv import lr_one_fold_measures_feature_selection

if __name__ == "__main__":
	spreadsheet = Spreadsheet(project_data_file)
	data = Data(spreadsheet)
	targets = data.targets
	ids = data.ids

	try:
		[dataset, features] = parse_theme(sys.argv[1])
		feature_selection_before(features, targets, dataset, CV_PERCENTAGE_OCCURENCE_THRESHOLD, ids, lr_one_fold_measures_feature_selection)
		
	except IndexError:
		print "Error!! Pass 'all' as argument"
