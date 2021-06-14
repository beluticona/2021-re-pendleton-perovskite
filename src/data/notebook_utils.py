from src.config import training_dataset_path, data_types_path, chemical_inventory_path
import src.data.utils as data_utils 
import pandas as pd
import json
import yaml as yl


def read_data(data_path=training_dataset_path):
    with open(data_types_path) as json_file:

        dtypes = json.load(json_file)
        
        df = pd.read_csv(data_path, header=0, dtype=dtypes)

        data_utils.select_experiment_version_and_used_solvent(df)

        df = df.fillna(0)

        df['_raw_reagent_5_chemicals_2_actual_amount'] = [0] * df.shape[0]

        return df


def read_chemical_info():
    with open(chemical_inventory_path) as file:
        chemical_inventory = pd.read_csv(file, header=0)
        return chemical_inventory


def filter_columns_by_prefix(columns, prefixes):
    filtered_columns = { column for column in columns if True in (column.startswith(prefix)  for prefix in prefixes) }
    return filtered_columns


def get_columns(total_columns):
    prefixs = ['_rxn_', '_feat_', '_raw']
    columns_by_prefix = {}
    for prefix in prefixs:
        columns_by_prefix[prefix] = set(filter(lambda column_name: column_name.startswith(prefix), total_columns))
    columns_by_prefix['solUD'] = set(data_utils.get_sol_ud_model_columns(total_columns))
    columns_by_prefix['solV'] = set(data_utils.get_sol_v_model_columns(total_columns))
    chem_cols = []
    data_utils.extend_with_chem_columns(total_columns, chem_cols)
    columns_by_prefix['chem'] = set(chem_cols)
    return columns_by_prefix


# We limit the scope only analizing solUD information, we could have used solV
def get_used_columns(total_columns):
    to_add = []
    data_utils.extend_with_rxn_columns(total_columns, to_add, sol_ud_enable=True)
    #data_utils.extend_with_rxn_columns(total_columns, to_add, sol_ud_enable=False)
    data_utils.extend_with_chem_columns(total_columns, to_add)

    #to_add += '_raw_v0-M_acid _raw_v0-M_inorganic _raw_v0-M_organic'.split(' ')
    return set(to_add)
