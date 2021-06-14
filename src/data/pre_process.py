
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


def prepare_full_dataset(df, data_preparation):

    """DECISION FUERTE, POR QUÉ NO CONSIDERAR LOS QUE TENGAN 3? Borra los que tienen 0
     Collect inches of solvents that had at least a successful crystal
    """

    successful_inches_keys = df.query('_out_crystalscore == 4')['_rxn_organic-inchikey'].unique()

    df = (df[df['_rxn_organic-inchikey'].isin(successful_inches_keys)])

    #df.query('_out_crystalscore > 0', inplace=True)

    # If it is desired to make a non sense dataset
    if data_preparation["shuffle_enabled"]:
        df = shuffle(df)
    elif data_preparation["deep_shuffle_enabled"]:
        df = deep_shuffle(df)

    # Rename from raw to rxn
    #df.rename(columns={"_raw_v0-M_acid": "_rxn_v0-M_acid", "_raw_v0-M_inorganic": "_rxn_v0-M_inorganic",
    #                   "_raw_v0-M_organic": "_rxn_v0-M_organic"}, inplace=True)

    return df


def get_predictors_types(dataset_name):
    type_sol = NO_MODEL
    if 'solV' in dataset_name:
        type_sol = SOLV_MODEL 
    if 'solUD' in dataset_name:
        type_sol = SOLUD_MODEL
    return type_sol, 'feat' in dataset_name, 'chem' in dataset_name, 'exp' in dataset_name, 'reag' in dataset_name


def process_dataset(df, parameters, full_results):
    interpolate, extrapolate = parameters['intrpl'], parameters['extrpl']
    inchis = df['_rxn_organic-inchikey']

    # Binary class
    crystal_score = df['_out_crystalscore']
    crystal_score = (crystal_score == 4).astype(int)
    df['_out_crystalscore'] = crystal_score

    # Preparing data
    if parameters['model']['strat']:
        #TODO: solo pasar df, tomar max cant de distintos inchikeys
        selected_data, crystal_score = data_utils.stratify(df, crystal_score, df['_rxn_organic-inchikey'].values)

    # Split train & test
    #X, X_test, y, y_test = train.train_test_split(df, crystal_score, test_size=0.2, random_state=parameters['model']['seed'])

    # Select predictors combinations to run  
    selected_predictors_combinations = [predictors_combination for (predictors_combination, required) in parameters["dataset"].items() if required]
    if parameters['fixed-predictors']:
        selected_predictors_combinations = ['selected-predictors']

    # for each combination, train and predict considering parameters
    for dataset_name in selected_predictors_combinations:
        if parameters['fixed-predictors']:
            selected_data = data_utils.filter_top_worst_cols(df, parameters)
        else:
            type_sol_volume, feat_extend_enabled, chem_extend_enabled, exp_extend_enabled, reag_extend_enabled = get_predictors_types(dataset_name)
            selected_data = data_utils.filter_required_data(df, type_sol_volume, feat_extend_enabled, chem_extend_enabled, exp_extend_enabled, reag_extend_enabled)

        if parameters['model']['one-hot-encoding']:
            selected_data = data_utils.encode_by_amine_inchi(df[['_rxn_organic-inchikey']], selected_data, df.columns)

        # Processing data
        if interpolate:
            train.std_train_test(selected_data, parameters['model'], crystal_score, dataset_name, full_results['std'])
        
        print('Complete: dataset '+ dataset_name + 'intrpl')

        if extrapolate:
            train.leave_one_out_train_test(selected_data, parameters['model'], crystal_score, dataset_name, inchis, full_results['loo'])

        print('Complete: dataset '+ dataset_name + 'all' + str(parameters['model']['sample_fraction']))
    


    '''
    TODO:
        - move filter data columns do data intead of train
            * data feat scaling will be there
        - break utils into
                * main: process_dataset
                * get_data: filters, prepare, shuffle


    '''

