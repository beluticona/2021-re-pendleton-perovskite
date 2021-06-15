from src.data import pre_process
from src.data import post_process
from src.config import training_dataset_path, parameters_path, data_types_path
from src.models import utils as model_utils
import pandas as pd
import yaml as yl
import json


with open(parameters_path) as file, open(data_types_path) as json_file:

    parameters = yl.safe_load(file)

    dtypes = json.load(json_file)
    
    df = pd.read_csv(training_dataset_path, header=0, dtype=dtypes)

    # Shuffle dataset if required (to compare results)
    df = pre_process.shuffle_dataset(df, parameters["data_preparation"])

    #To analize metrics vs amount of data
    #sampling_fractions = [round(0.1*n,2) for n in range(2,11)]
    sampling_fractions = [1]
    full_results = model_utils.create_results_container(parameters)

    # Set seed for each run
    random_seeds = [random_seed for random_seed in range(parameters['runs'])]

    # Read classifeirs to run

    for seed in random_seeds:
        for fraction in sampling_fractions:
            frac_df = df.groupby('_out_crystalscore').sample(frac=fraction, random_state=seed)
            parameters['model']['sample_fraction'] = fraction 
            parameters['model']['seed'] = seed 
            pre_process.process_dataset(frac_df, parameters, full_results) 

    post_process.save_results(full_results, parameters)

