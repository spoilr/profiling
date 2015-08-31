#!/usr/bin/python

import numpy as np
import xlrd
import themes

FEATURE_ROW = 0
START_EXAMPLE_ROW = 1

class Data:

	def __init__(self, spreadsheet, upsampling=True):
		self.upsampling = upsampling
		self.spreadsheet = spreadsheet
		self.features = spreadsheet.features
		self.targets = spreadsheet.targets
		self.examples = self.extract_examples(spreadsheet)
		self.selected_features = themes.subset_features
		self.network_features = themes.network
		self.ideology_features = themes.ideology
		self.illness_features = themes.illness
		self.ids = spreadsheet.ids

	# Return as np array
	def extract_examples(self, spreadsheet):
		return spreadsheet.examples

	def extract_network_indices(self):
		network_indices = self.extract_indices_from_themes(self.network_features)
		assert len(network_indices) == len(self.network_features)
		return network_indices

	def extract_ideology_indices(self):
		ideology_indices = self.extract_indices_from_themes(self.ideology_features)
		assert len(ideology_indices) == len(self.ideology_features)
		return ideology_indices
		
	def extract_illness_indices(self):
		illness_indices = self.extract_indices_from_themes(self.illness_features)		
		assert len(illness_indices) == len(self.illness_features)
		return illness_indices

	def extract_selected_indices(self):
		selected_indices = self.extract_indices_from_themes(self.selected_features)
		assert len(selected_indices) == len(self.selected_features)
		return selected_indices

	def extract_indices_from_themes(self, thematic_features):
		thematic_indices = []

		for x in thematic_features:
			try:
				thematic_indices.append(self.features.index(x))
			except ValueError:
				print "Theme %s is not in the Main Features!" % x	

		return thematic_indices	

	# Return as np array
	def extract_examples_with_features_from_indices(self, indices):
		num_rows = self.spreadsheet.worksheet.nrows
		thematic_examples = []

		for curr_row in range(START_EXAMPLE_ROW, num_rows):
			row = []
			for ind in indices:
				row.append(self.spreadsheet.get_cell(curr_row, ind))

			thematic_examples.append(row)	

			if self.upsampling:
				if self.targets[curr_row-1] == 1:
					row = []
					for ind in indices:
						row.append(self.spreadsheet.get_cell(curr_row, ind))				
					thematic_examples.append(row)		

		assert len(thematic_examples) == len(self.examples)
		
		return np.asarray(thematic_examples)

	def extract_network_examples(self):
		return self.extract_examples_with_features_from_indices(self.extract_network_indices())	

	def extract_ideology_examples(self):
		return self.extract_examples_with_features_from_indices(self.extract_ideology_indices())
		
	def extract_illness_examples(self):
		return self.extract_examples_with_features_from_indices(self.extract_illness_indices())

	def extract_selected_examples(self):
		return self.extract_examples_with_features_from_indices(self.extract_selected_indices())	


class Spreadsheet:

	SHEET = 'Sheet1'

	def __init__(self, url, upsampling=True):
		self.upsampling = upsampling
		self.workbook = xlrd.open_workbook(url)
		self.worksheet = self.workbook.sheet_by_name(self.SHEET)
		self.features = self.get_features()
		self.examples = self.get_examples()
		self.targets = self.get_targets()

		if self.upsampling:
			for i in range(len(self.targets)):
				if self.targets[i] == 1:
					self.examples = np.vstack((self.examples, self.examples[i]))
					self.targets = np.hstack((self.targets, self.targets[i]))

		
		self.ids = self.get_ids()

	def get_ids(self):
		ids = []
		num_rows = self.worksheet.nrows
		for curr_row in range(START_EXAMPLE_ROW, num_rows):
			ids.append(self.get_cell(curr_row, 0))

		if self.upsampling:
			for curr_row in range(START_EXAMPLE_ROW, num_rows):
				if self.targets[curr_row-1] == 1:
					ids.append(self.get_cell(curr_row, 0))	
			
		return ids	

	def get_features(self):
		return self.get_cell_names()[:-1]

	def get_targets(self):
		targets = []
		num_rows = self.worksheet.nrows
		target_column_index	= self.worksheet.ncols - 1
		for row in range(1, num_rows): # ignore the feature name
			targets.append([self.get_cell(row, target_column_index)])
		targets = np.array(targets)
		return [int(x) for x in targets[:,0]] # convert strings to floats 

	def get_cell_names(self):
		return self.get_row(FEATURE_ROW)

	def get_examples(self):
		num_rows = self.worksheet.nrows
		examples = []

		for curr_row in range(START_EXAMPLE_ROW, num_rows):
			examples.append(self.get_row(curr_row))

		np_examples = np.asarray(examples)	
		target_column_index	= np_examples.shape[1] - 1
		return np.delete(np_examples, [target_column_index], 1) # remove the target column from the example array

	def get_row(self, curr_row):
		data_row = []
		row = self.worksheet.row(curr_row)
		num_cols = self.worksheet.ncols
		for curr_cell in range(0, num_cols):
			cell_value = self.worksheet.cell_value(curr_row, curr_cell)
			data_row.append(cell_value)
		return data_row

	def get_cell(self, curr_row, curr_cell):
		return self.worksheet.cell_value(curr_row, curr_cell)

	
if __name__ == "__main__":
	spreadsheet = Spreadsheet('../../Downloads/ip/project data.xlsx')
	data = Data(spreadsheet)