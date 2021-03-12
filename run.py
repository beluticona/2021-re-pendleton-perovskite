from src.data import utils
from src.config import data_path
import pandas as pd
import yaml as yl


with open('parameters.yaml') as file:
	parameters = yl.safe_load(file)
	
	df = pd.read_csv(data_path)

	# Select reactions where the solvent produced at leat a crystal score of 4
	succesful_reaction = utils.maka_dataset(df)

	if parameters["shuffle"]: utils.shuffle(df)
	elif parameters["deep_shuffle"]: utils.deep_shuffle(df)


	print(df.columns)

	#successful_perov.rename(columns={"_raw_v0-M_acid": "_rxn_v0-M_acid", "_raw_v0-M_inorganic": "_rxn_v0-M_inorganic", "_raw_v0-M_organic":"_rxn_v0-M_organic"}, inplace=True)



	'''
	- generate results
		* from yaml
			- classifiers
			- std norm ...

	

	'''