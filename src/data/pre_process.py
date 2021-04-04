from src.data import utils
from src.models import train
from src import constants
import numpy as np
import pandas as pd

experiment_version = 1.1

GBL_inchi_key = 'YEJRWHAVMIAJKC-UHFFFAOYSA-N'

dimethyl_ammonium_inchi_key = 'MXLWMIFDJCGBV-UHFFFAOYSA-N'


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
    # Select data from version 1.1
    df.query('_raw_ExpVer == @experiment_version', inplace=True)

    # Select reactions where only GBL is used as solvent 
    df.query('_raw_reagent_0_chemicals_0_InChIKey == @GBL_inchi_key', inplace=True)

    # Remove some anomalous entries with dimethyl ammonium still listed as the organic. 
    df.query('_raw_reagent_0_chemicals_0_InChIKey != @dimethyl_ammonium_inchi_key', inplace=True)

    """DECISION FUERTE, POR QUÉ NO CONSIDERAR LOS QUE TENGAN 3? Borra los que tienen 0
     Collect inches of solvents that had at least a successful crystal
    """

    successful_inches_keys = df.query('_out_crystalscore == 4')['_rxn_organic-inchikey'].unique()

    df = (df[df['_rxn_organic-inchikey'].isin(successful_inches_keys)])

    df.query('_out_crystalscore > 0', inplace=True)

    # If it is desired to make a non sense dataset
    if data_preparation["shuffle_enabled"]:
        df = shuffle(df)
    elif data_preparation["deep_shuffle_enabled"]:
        df = deep_shuffle(df)

    # Rename from raw to rxn
    df.rename(columns={"_raw_v0-M_acid": "_rxn_v0-M_acid", "_raw_v0-M_inorganic": "_rxn_v0-M_inorganic",
                       "_raw_v0-M_organic": "_rxn_v0-M_organic"}, inplace=True)

    return df


def detect_type_dataset(dataset_name):
    if 'solV' in dataset_name:
        return constants.SOLV_MODEL, 'chem' in dataset_name, 'exp' in dataset_name, 'reag' in dataset_name
    if 'solUD' in dataset_name:
        return constants.SOLUD_MODEL, 'chem' in dataset_name, 'exp' in dataset_name, 'reag' in dataset_name


def process_dataset(df, parameters):
    # inchis = df['_rxn_organic-inchikey'].value_counts()

    # binary class
    crystal_score = df['_out_crystalscore']
    crystal_score = (crystal_score == 4).astype(int)

    # extr = parameters["extrpl"]
    interpolate = parameters["intrpl"]

    results = {
        'dataset_index': [],
        'cv': [],
        #    'matrix':[],
        'precision_positive': [],
        'recall_positive': [],
        'f1_positive': [],
        'support_negative': [],
        'support_positive': [],
        'matthewCoef': []
    }

    requested_datasets = [dataset_name for (dataset_name, required) in parameters["dataset"].items() if required]

    # for each dataset, train and predict considering parameters
    for dataset_name in requested_datasets:
        type_sol_volume, chem_extend_enabled, exp_extend_enabled, reag_extend_enabled = detect_type_dataset(dataset_name)
        selected_data = utils.filter_required_data(df, type_sol_volume, chem_extend_enabled, exp_extend_enabled, reag_extend_enabled)

        # stratify crystal score out of loop
        if parameters['model']['strat']:
            selected_data, crystal_score = utils.stratify(selected_data, crystal_score, df['_rxn_organic-inchikey'].values)

        if parameters['model']['one-hot-encoding']:
            selected_data = utils.encode_by_amine_inchi(df[['_rxn_organic-inchikey']], selected_data, df.columns)

        if interpolate:
            train.std_train_test(selected_data, parameters["model"], crystal_score, dataset_name, results)


    # save results 
    return results


    '''
    TODO:
        - move filter data columns do data intead of train
            * data feat scaling will be there
        - break utils into
                * main: process_dataset
                * get_data: filters, prepare, shuffle


    '''
