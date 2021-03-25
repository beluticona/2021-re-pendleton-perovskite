from src.data import process
from src.config import data_path
from src.config import data_types_path
from src.config import results_path
from src.config import file_name_for
import pandas as pd
import yaml as yl
import json
import numpy as np 


with open('parameters.yaml') as file, open(data_types_path) as json_file:

    parameters = yl.safe_load(file)

    dtypes = json.load(json_file)
    
    df = pd.read_csv(data_path, header=0, dtype=dtypes)

    # Select reactions where the solvent produced at least a crystal score of 4
    df = process.prepare_full_dataset(df, parameters["data_preparation"])

    results = process.process_dataset(df, parameters)

    results.to_csv(results_path + file_name_for(parameters)[:-1] + '.csv')
    '''
    - generate results
        * from yaml
            - classifiers
            - std norm ...

    # fijar semilla
    

    '''
    # print(file_name_for(parameters)[:-1])
