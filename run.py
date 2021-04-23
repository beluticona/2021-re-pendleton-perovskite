from src.data import pre_process
from src.data import post_process
from src.config import data_path, parameters_path, data_types_path
import pandas as pd
import yaml as yl
import json


with open(parameters_path) as file, open(data_types_path) as json_file:

    parameters = yl.safe_load(file)

    dtypes = json.load(json_file)
    
    df = pd.read_csv(data_path, header=0, dtype=dtypes)

    # Select reactions where the solvent produced at least a crystal score of 4
    df = pre_process.prepare_full_dataset(df, parameters["data_preparation"])

    pre_process.process_dataset(df, parameters)


    '''
    # fijar semilla global
    

    '''
    # print(file_name_for(parameters)[:-1])
