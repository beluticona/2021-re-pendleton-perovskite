from sklearn.preprocessing import OneHotEncoder
from src import constants
from src.constants import experiment_version, GBL_inchi_key, dimethyl_ammonium_inchi_key
import pandas as pd
import numpy as np
import re


def get_sol_ud_model_columns(df_columns):
    sol_ud_model_columns = list(filter(lambda column_name: column_name.startswith('_rxn_') and not column_name.startswith('_rxn_v0'), df_columns))
    sol_ud_model_columns.remove('_rxn_organic-inchikey')
    return sol_ud_model_columns


def get_physicochemical_properties_columns(df_columns, subset_columns_to_extend):
    feat_columns_to_add = list(filter(lambda column_name: column_name.startswith('_feat_'), df_columns))
    return feat_columns_to_add + subset_columns_to_extend   
    # @TODO: _feat_vanderwalls_volume no filtra nada, solV ya lo incluye por ser _featVanderWaalsVolme


def filter_required_data(df, dataset_name):
    df_columns = df.columns.to_list()
    selected_predictors = []
    if constants.SOLUD_MODEL in dataset_name:
        selected_predictors += get_sol_ud_model_columns(df_columns)
    if constants.PHYSICOCHEMICAL in dataset_name:
        selected_predictors += get_physicochemical_properties_columns(df_columns, selected_predictors)
    
    return df[selected_predictors].fillna(0).reset_index(drop=True)


def filter_top_worst_cols(df, parameters):
    cols = parameters['col_num_selected']
    selected_features = constants.features_sorted_by_importance[:cols]
    if parameters['top-tail'] == 0:
        tail = len(constants.features_sorted_by_importance)-3*cols
        selected_features = constants.features_sorted_by_importance[cols*2:-tail]
    return df[selected_features].fillna(0).reset_index(drop=True)


def select_experiment_version_and_used_solvent(df):
    df.query('_raw_ExpVer == @experiment_version', inplace=True)

    # Select reactions where only GBL is used as solvent 
    df.query('_raw_reagent_0_chemicals_0_InChIKey == @GBL_inchi_key', inplace=True)

    # Remove some anomalous entries with dimethyl ammonium still listed as the organic. 
    df.query('_raw_reagent_0_chemicals_0_InChIKey != @dimethyl_ammonium_inchi_key', inplace=True)
