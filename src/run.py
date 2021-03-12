import yaml as yl
from src.data import utils


with open('parameters.yaml') as file:
	parameters = yl.safe_load(file)
	print(parameters)

	print(utils.make_dataset)




	# import yaml parameters
		# - models
		# - param grid
	


	'''
	- generate results
		* from yaml
			- classifiers
			- std norm ...

	

	'''