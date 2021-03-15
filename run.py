from src.data import utils
from src.models import train_model
from src.config import data_path
from src.config import data_types_path
import pandas as pd
import yaml as yl
import json
import numpy as np 


with open('parameters.yaml') as file, open(data_types_path) as json_file:

	parameters = yl.safe_load(file)
    
	dtypes = json.load(json_file)	
	
	df = pd.read_csv(data_path,header=0, dtype = dtypes)

	# Select reactions where the solvent produced at leat a crystal score of 4
	df = utils.prepare_dataset(df, parameters["shuffle"], parameters["deep_shuffle"])

	# for each dataset, train and predict considering parameters
	
	##----------------------TO TEST--------------------------------##
	# Count incinchikeys
	inchikeys_count = df['_rxn_organic-inchikey'].value_counts()
	
	# binarize class
	crystal_score = df['_out_crystalscore']
	crystal_score = (out == 4).astype(int)

	for dataset_name in parameters["dataset"]: 
		utils.process_dataset(df, dataset_name, parameters)



	'''
	- generate results
		* from yaml
			- classifiers
			- std norm ...

	

	'''