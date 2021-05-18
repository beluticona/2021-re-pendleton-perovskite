from src.config import data_path, parameters_path, data_types_path
import src.data.utils as data_utils 
import pandas as pd
import json
import yaml as yl


def read_data():
    with open(parameters_path) as file, open(data_types_path) as json_file:

        parameters = yl.safe_load(file)

        dtypes = json.load(json_file)
        
        df = pd.read_csv(data_path, header=0, dtype=dtypes)

        data_utils.select_experiment_version_and_used_solvent(df)

        df = df.fillna(0)

        df['_raw_reagent_5_chemicals_2_actual_amount'] = [0] * df.shape[0]

        return df


def get_columns(total_columns):
    prefixs = ['_rxn_', '_feat_']
    columns_by_prefix = {}
    for prefix in prefixs:
        columns_by_prefix[prefix] = set(filter(lambda column_name: column_name.startswith(prefix), total_columns))
    columns_by_prefix['solUD'] = set(data_utils.get_sol_ud_model_columns(total_columns))
    chem_cols = []
    data_utils.extend_with_chem_columns(total_columns, chem_cols)
    columns_by_prefix['chem'] = chem_cols
    return columns_by_prefix


# We limit the scope only analizing solUD information, we could have used solV
def get_used_columns(total_columns):
    to_add = []
    data_utils.extend_with_rxn_columns(total_columns, to_add, sol_ud_enable=True)
    #data_utils.extend_with_rxn_columns(total_columns, to_add, sol_ud_enable=False)
    data_utils.extend_with_chem_columns(total_columns, to_add)

    #to_add += '_raw_v0-M_acid _raw_v0-M_inorganic _raw_v0-M_organic'.split(' ')
    return set(to_add)
