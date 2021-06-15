
from src.data import utils as data_utils
from src.data import post_process
from src.models import train
from src.models import utils as model_utils
from src.constants import SOLV_MODEL, SOLUD_MODEL, NO_MODEL
import numpy as np
import pandas as pd


def shuffle(df):
    """Generate a non sense data set:
    Shuffle all row except by crystal score column
    """
    out_hold = df['_out_crystalscore']
    df = df.reindex(np.random.permutation(df.index)).reset_index(drop=True)
    df['_out_crystalscore'] = out_hold
    return df


def deep_shuffle(df):
    # Not shuffled: raw
    kept_columns = df.loc[:, '_raw_model_predicted':'_prototype_heteroatomINT']
    kept_columns = pd.concat([df['_rxn_organic-inchikey'], kept_columns], axis=1)

    # Isolated shuffle: all but raw
    shuffle_rxn = pd.concat([df.loc[:, 'name':'_rxn_M_organic'],
                             df.loc[:, '_rxn_temperatureC_actual_bulk': '_feat_Hacceptorcount']],
                            axis=1)
    shuffled_rxn = shuffle_rxn.apply(np.random.permutation).reset_index(drop=True)

    shuffled_reactions = pd.concat([kept_columns, shuffled_rxn], axis=1)

    return shuffled_reactions


def shuffle_dataset(df, data_preparation):
    # If it is desired to make a non sense dataset
    if data_preparation["shuffle_enabled"]:
        df = shuffle(df)
    elif data_preparation["deep_shuffle_enabled"]:
        df = deep_shuffle(df)

    return df


def process_dataset(df, parameters, full_results):
    # Binary class
    crystal_score = df['_out_crystalscore']
    crystal_score = (crystal_score == 4).astype(int)
    df['_out_crystalscore'] = crystal_score

    # Select predictors combinations to run: RXN | CHEMICAL | union(RXN+CHEMICAL)  
    selected_predictors_combinations = [predictors_combination for (predictors_combination, required) in parameters["dataset"].items() if required]
    
    if parameters['fixed-predictors']:
        selected_predictors_combinations = ['selected-predictors']

    # for each combination, train and predict considering parameters
    for dataset_name in selected_predictors_combinations:
        if parameters['fixed-predictors']:
            selected_data = data_utils.filter_top_worst_cols(df, parameters)
        else:
            selected_data = data_utils.filter_required_data(df, dataset_name)

        # Processing data
        train.std_train_test(selected_data, parameters['model'], crystal_score, dataset_name, full_results, )
        
        print('Complete: dataset '+ dataset_name + str(parameters['model']['sample_fraction']))
    

    '''
    TODO:
        - move filter data columns do data intead of train
            * data feat scaling will be there
        - break utils into
                * main: process_dataset
                * get_data: filters, prepare, shuffle


    '''

