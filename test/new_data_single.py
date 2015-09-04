import sys
sys.path.insert(0, 'utils/')
sys.path.insert(0, 'classification/')
from load_data import *
from project_data import *
from parse_theme import *
from standardized_data import *
from cv import single_svm_one_fold_measures
from cv import lr_one_fold_measures
from cv import dt_one_fold_measures
from cv import knn_one_fold_measures
from replace_missing_values import *
from save_output import save_output

import numpy as np
from sklearn.preprocessing import StandardScaler

def new_data_single(training_data, training_targets, testing_data, testing_targets, tech):
	[training_data, training_targets, unk] = split_dataset(training_data, training_targets)
		
	# standardize dataset - Gaussian with zero mean and unit variance
	scaler = StandardScaler()

	testing_data = replace_missings(testing_data)

	if tech == 'lr':
		error_rate, f1, model, (hp, hr, hf), (cp, cr, cf) = lr_one_fold_measures(training_data, testing_data, training_targets, testing_targets)

	elif tech == 'dt':
		error_rate, f1, model, (hp, hr, hf), (cp, cr, cf) = dt_one_fold_measures(training_data, testing_data, training_targets, testing_targets)
		
	elif tech == 'knn':
		training_data = scaler.fit_transform(training_data)
		testing_data = scaler.transform(testing_data)

		error_rate, f1, model, (hp, hr, hf), (cp, cr, cf) = knn_one_fold_measures(training_data, testing_data, training_targets, testing_targets)

	elif tech == 'svm':
		training_data = scaler.fit_transform(training_data)
		testing_data = scaler.transform(testing_data)

		error_rate, f1, model, (hp, hr, hf), (cp, cr, cf) = single_svm_one_fold_measures(training_data, testing_data, training_targets, testing_targets)

	else:
		print 'ERROR technique'	

	print 'Final error %f' % error_rate
	print 'Final accuracy %f' % (1 - error_rate)

	print 'Highval precision %f' % hp
	print 'Highval recall %f' % hr
	print 'Highval f1 %f' % hf
	print 'Civil precision %f' % cp
	print 'Civil recall %f' % cr
	print 'Civil f1 %f' % cf

	return error_rate, f1, model, (hp, hr, hf), (cp, cr, cf)

if __name__ == "__main__":

	training_spreadsheet = Spreadsheet(project_data_file)
	training_data = Data(training_spreadsheet)
	training_targets = training_data.targets

	testing_spreadsheet = Spreadsheet(addendum_data_file, upsampling=False)
	testing_data = Data(testing_spreadsheet, upsampling=False)
	testing_targets = testing_data.targets

	[training_data, features] = parse_theme('all')
	[testing_data, feats] = parse_theme_from_file('all', addendum_data_file)
	assert features == feats

	tech = raw_input("Enter algorithm. Choose between lr, dt, knn, svm")

	file_name = "new_single_" + tech + ".txt"
	for i in range(100):
		error_rate, f1, model, (hp, hr, hf), (cp, cr, cf) = new_data_single(training_data, training_targets, testing_data, testing_targets, tech)
		save_output(file_name, error_rate, hp, hr, hf, cp, cr, cf, 1)

	

