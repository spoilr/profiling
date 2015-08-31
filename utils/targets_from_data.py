from load_data import *
from project_data import *
import numpy as np

target = 'Target'
target_group = 'TargetGroup'

def extract_target(features):
	return features.index(target)

def extract_target_group(features):
	return features.index(target_group)


if __name__ == "__main__":
	spreadsheet = Spreadsheet(project_data_file)
	data = Data(spreadsheet)	
	target_index = extract_target(data.features)
	target_group_index = extract_target_group(data.features)

	np_examples = np.array(data.examples)
	targets = np_examples[:, target_index]
	target_groups = np_examples[:, target_group_index]

	target_set = set(targets)
	for s in target_set:
		print s