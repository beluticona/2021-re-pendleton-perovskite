import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.preprocessing import Normalizer, StandardScaler
from src import constants


def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]


def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]


def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]


def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]


def mcc(y_true, y_pred): return matthews_corrcoef(y_true, y_pred)


def sup1(y_true, y_pred): return np.sum(y_true)


def sup0(y_true, y_pred): return len(y_true) - np.sum(y_true)


def columns_to_scale(column_list, std_dict, norm_dict):
    curated_list = []
    for header_prefix in std_dict:
        if std_dict[header_prefix] == 1:
            for column in column_list:
                if header_prefix in column:
                    curated_list.append(column)
    for header_prefix in norm_dict:
        if norm_dict[header_prefix] == 1:
            for column in column_list:
                if header_prefix in column:
                    curated_list.append(column)
    return curated_list


def feat_scaling(model_parameters, data_columns):
    requested_norm = [dataset_name for (dataset_name, required) in model_parameters["norm"].items() if required]
    requested_sdt = [dataset_name for (dataset_name, required) in model_parameters["std"].items() if required]

    if len(requested_norm) + len(requested_sdt) == 0:
        return None, []
    else:
        curated_columns = columns_to_scale(data_columns, model_parameters['std'], model_parameters['norm'])
        if len(requested_norm) > 0: fun = Normalizer()
        else: fun = StandardScaler()

    return make_column_transformer((fun, curated_columns), remainder='passthrough'), curated_columns


def translate_inchi_key(inchi, results):
    chemical_name = constants.INCHI_TO_CHEMNAME[inchi]
    results['chemical-name'].append(chemical_name)


def no_feat_scaling(model_parameters):
    std = sum(value for value in model_parameters['std'].values())
    norm = sum(value for value in model_parameters['norm'].values())
    return std + norm == 0


def create_results_container():
    return result_container()

def result_container():
    std_results = {
        'data_index': [],
        'sample_fraction': [],
        'seed': [],
        'cv': [],
        'precision_positive': [],
        'recall_positive': [],
        'f1_positive': [],
        'support_negative': [],
        'support_positive': [],
        'matthewCoef': []
    }
    return std_results
