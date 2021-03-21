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
    df = utils.prepare_dataset(df, parameters["data_preparation"])

    utils.process_dataset(df, parameters)

    '''
    - generate results
        * from yaml
            - classifiers
            - std norm ...

    # fijar semilla
    

    '''