from src.models import train
import numpy as np
import pandas as pd


experiment_version = 1.1

GBL_inchikey = 'YEJRWHAVMIAJKC-UHFFFAOYSA-N'

dimethyl_ammonium_inchikey = 'MXLWMIFDJCGBV-UHFFFAOYSA-N'

'''Generate a non sense data set:
Shuffle all row except by crystal score column'''
def shuffle(df):
    out_hold = df['_out_crystalscore']
    df = df.reindex(np.random.permutation(df.index)).reset_index(drop=True)
    df['_out_crystalscore'] = out_hold
    return df

def deep_shuffle(df):
    # Not shuffled: raw
    keeped_columns = df.loc[:, '_raw_model_predicted':'_prototype_heteroatomINT']
    keeped_columns = pd.concat([df['_rxn_organic-inchikey'], keeped_columns], axis=1) 

    # Isolated shuffle: all but raw
    shuffle_rxn = pd.concat([df.loc[:, 'name':'_rxn_M_organic'], 
                            df.loc[:, '_rxn_temperatureC_actual_bulk' : '_feat_Hacceptorcount']], 
                            axis = 1)
    shuffled_rxn = shuffle_rxn.apply(np.random.permutation).reset_index(drop=True)

    shuffled_reactions = pd.concat([keeped_columns, shuffled_rxn], axis=1)

    return shuffled_reactions

def prepare_dataset(df, data_preparation):
    
    # Select data from version 1.1
    df.query('_raw_ExpVer == @experiment_version', inplace = True)

    # Select reactions where only GBL is used as solvent 
    df.query('_raw_reagent_0_chemicals_0_InChIKey == @GBL_inchikey', inplace = True)
    
    # Remove some anomalous entries with dimethyl ammonium still listed as the organic. 
    df.query('_raw_reagent_0_chemicals_0_InChIKey != @dimethyl_ammonium_inchikey', inplace = True) 
    
    ## DECISION FUERTE, POR QUÉ NO CONSIDERAR LOS QUE TENGAN 3?
    # Collect inchikeys of solvents that had at least a succesful crystal
    
    succesful_inchikeys = df.query('_out_crystalscore == 4')['_rxn_organic-inchikey'].unique()
    
    df = (df[df['_rxn_organic-inchikey'].isin(succesful_inchikeys)])
    
    # If it is desired to make a non sence dataset
    if data_preparation["shuffle_enabled"]: 
        df = shuffle(df)
    elif data_preparation["deep_shuffle_enabled"]: 
        df = deep_shuffle(df)

    # Rename from raw to rxn
    df.rename(columns={"_raw_v0-M_acid": "_rxn_v0-M_acid", "_raw_v0-M_inorganic": "_rxn_v0-M_inorganic", "_raw_v0-M_organic":"_rxn_v0-M_organic"}, inplace=True)

    return df


##-----------------------------esqueleto-------------##

def process_dataset(df, parameters):
    inchikeys_count = df['_rxn_organic-inchikey'].value_counts()

    # binarize class
    crystal_score = df['_out_crystalscore']
    crystal_score = (crystal_score == 4).astype(int)

    extr = parameters["extrpl"]
    intrpl = parameters["intrpl"]

    results = {
        'dataset_index': [],
        'matrix':[],
        'precision_positive':[],
        'recall_positive':[],
        'f1_positive':[],
        'support_negative':[],
        'support_positive':[],
        'matthewCoef':[]
    }

    # for each dataset, train and predict considering parameters
    for dataset_name in parameters["dataset"]: 

        selected_data = {    
            'solV-chem': train.filter_data_for_solV(df),
            'solUD-chem': train.filter_data_for_solUD(df)
        }[dataset_name]

        if(intrpl): train.std_train_test(selected_data, parameters["model"], crystal_score, dataset_name, results)
       
    # save results 
    df =pd.DataFrame.from_dict(results, orient='columns')
    df.to_csv('test.csv')

    '''
    TODO:
        - move filter data columns do data intead of train
            * data feat scaling will be there
        - break utils into
                * main: process_dataset
                * get_data: filters, prepare, shuffle


    '''
 