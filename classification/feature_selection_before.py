import sys
sys.path.insert(0, 'results/')
from feature_selection_cv import *
from cv import cv10
from cv import cv10_ensemble
from features_from_svm_selection import single_features_90
from features_from_svm_selection import single_features_70
from features_from_svm_selection import single_features_50

import numpy as np

def feature_selection_before_ensemble(features, targets, dataset, percentage, dt, knn, svm, prt=False, file_name=None):
	[known_dataset, known_targets, unk] = split_dataset(dataset, targets)
		
	known_targets = np.asarray(known_targets)

	# these come from feature_selection_cv
	# commented out because they were saved to decrease computation time
	# cv_features = features_cross_validation(known_dataset, known_targets, features)
	# selected_features = select_final_features_from_cv(cv_features, percentage)
	selected_features = select_features(percentage)

	sf = SelectedFeatures(known_dataset, known_targets, selected_features, features)
	known_dataset = sf.extract_data_from_selected_features()

	std = StandardizedData(known_targets, known_dataset)
	known_dataset_scaled, known_targets = std.split_and_standardize_dataset()  

	cv10_ensemble(np.array(known_dataset), np.array(known_targets), known_dataset_scaled, dt, knn, svm, prt, file_name)

	print '####### FEATURES ####### %d \n %s' % (len(selected_features), str(selected_features))

def feature_selection_before(features, targets, dataset, percentage, ids, one_fold_measures, standardize=False, prt=False, file_name=None):
	[known_dataset, known_targets, unk] = split_dataset(dataset, targets)
		
	known_targets = np.asarray(known_targets)

	# these come from feature_selection_cv
	# commented out because they were saved to decrease computation time
	# cv_features = features_cross_validation(known_dataset, known_targets, features)
	# selected_features = select_final_features_from_cv(cv_features, percentage)
	selected_features = select_features(percentage)

	sf = SelectedFeatures(known_dataset, known_targets, selected_features, features)
	known_dataset = sf.extract_data_from_selected_features()

	if standardize:
		std = StandardizedData(known_targets, known_dataset)
		known_dataset, known_targets = std.split_and_standardize_dataset()  

	cv10(np.array(known_dataset), known_targets, ids, one_fold_measures, prt, file_name)

	print '####### FEATURES ####### %d \n %s' % (len(selected_features), str(selected_features))


def select_features(percentage):
	if percentage == 0.9:
		return single_features_90
	elif percentage == 0.7:
		return single_features_70
	elif percentage == 0.5:
		return single_features_50
	else:
		print 'ERROR in percentage'			
