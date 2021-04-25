from src.config import file_name_for
from src.config import results_path
from src.config import set_results_folder
from src.constants import GBC, KNN, FEAT_NAMES_IMPORTANCE, FEAT_VALUES_IMPORTANCE, STD_CV, LOO
import pandas as pd
import collections


def save_results(full_results, parameters):
    full_results_folder_path = results_path + folder_for(parameters)
    set_results_folder(full_results_folder_path)
    if parameters['model']['method'] == GBC:
        save_feature_importance(full_results_folder_path, full_results['std'])
    if parameters['intrpl']:
        save_full_and_summary_for(full_results['std'], full_results_folder_path, 'StandardTestTrain_Full.csv', 'StandardTestTrain_Summary.csv', STD_CV)
    if parameters['extrpl']:
        save_full_and_summary_for(full_results['loo'], full_results_folder_path, 'LeaveOneOut_Full.csv', 'LeaveOneOut_Summary.csv', LOO)

    full_results_description = file_name_for(parameters)[:-1] + '.csv'
    f = open(full_results_folder_path + 'info.txt', "a")
    f.write(full_results_description)
    f.close()


def save_full_and_summary_for(dict_results, full_results_path, full_result_filename, summary_filename, validation):
    df_results = pd.DataFrame.from_dict(dict_results, orient='columns')
    df_results.to_csv(full_results_path + full_result_filename, index=None)

    summarize_results(df_results, validation)

    std = df_results.groupby(['data_index', 'sample_fraction']).std().add_suffix(' std')
    min_results = df_results.groupby(['data_index', 'sample_fraction']).min().add_suffix(' min')
    max_results = df_results.groupby(['data_index', 'sample_fraction']).max().add_suffix(' max')
    summary = df_results.groupby(['data_index', 'sample_fraction']).mean().add_suffix(' mean')

    summary = summary.join(std).join(min_results).join(max_results)

    summary.to_csv(full_results_path + summary_filename)


def save_feature_importance(full_results_folder_path, dict_results):
    df_results = pd.DataFrame.from_dict(dict_results, orient='columns')
    features = df_results[[FEAT_VALUES_IMPORTANCE, FEAT_NAMES_IMPORTANCE]]
    features.to_csv(full_results_folder_path + 'features.csv')
    dict_results.pop(FEAT_VALUES_IMPORTANCE, None)
    dict_results.pop(FEAT_NAMES_IMPORTANCE, None)


def summarize_results(results_df, validation):
    # to delete for both cases: STD_CV and LOO
    columns_name_to_delete = ['cv', 'support_negative', 'support_positive']
    if validation == LOO:
        columns_name_to_delete += ['chemical-name', 'inchi']

    results_df.drop(columns_name_to_delete, axis=1, inplace=True)


def folder_for(parameters):
    folder = ''
    method = parameters['model']['method']
    if method == KNN:
        folder += 'knn'
    elif method == GBC:
        folder += 'gbc'
    if not parameters['model']['hyperparam-opt']:
        folder += '1'
    norm = sum(value for value in parameters['model']['norm'].values())
    std = sum(value for value in parameters['model']['std'].values())
    if std > 0:
        folder += '_std' + str(std-1)
    if norm > 0:
        folder += '_norm' + str(norm-1)
    if parameters['model']['one-hot-encoding']:
        folder += '_hot_encode'
    if parameters['data_preparation']['deep_shuffle_enabled'] or parameters['data_preparation']['deep_shuffle_enabled']:
        folder += '_shuffle'
    return folder + '/'
