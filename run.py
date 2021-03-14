from src.data import utils
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
	df = utils.make_dataset(df)

	# If it is desired to non sence dataset
	if parameters["shuffle"]: 
		df = utils.shuffle(df)
	elif parameters["deep_shuffle"]: 
		df = utils.deep_shuffle(df)

	df.rename(columns={"_raw_v0-M_acid": "_rxn_v0-M_acid", "_raw_v0-M_inorganic": "_rxn_v0-M_inorganic", "_raw_v0-M_organic":"_rxn_v0-M_organic"}, inplace=True)



	'''
	- generate results
		* from yaml
			- classifiers
			- std norm ...

	

	'''