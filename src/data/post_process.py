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
        pass
        # @TODO: REFACTOR
        #save_feature_importance(full_results_description, full_results, FEAT_VALUES_IMPORTANCE)
        #save_feature_importance(full_results_description, full_results, FEAT_NAMES_IMPORTANCE)
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

    std = df_results.groupby('data_index').std().add_suffix(' std')
    min_results = df_results.groupby('data_index').min().add_suffix(' min')
    max_results = df_results.groupby('data_index').max().add_suffix(' max')
    summary = df_results.groupby('data_index').mean().add_suffix(' mean')

    summary = summary.join(std).join(min_results).join(max_results)

    summary.to_csv(full_results_path + summary_filename)


def save_feature_importance(full_results_file_name, results, feature):
    pre_multilevel_dictionary = collections.defaultdict(dict)
    for i in range(len(results['data_index'])):
        pre_multilevel_dictionary[results['data_index'][i]][results['cv'][i]] = results[feature][i]
    multilevel_dictionary = {}
    for outerKey, innerDict in pre_multilevel_dictionary.items():
        for innerKey, values in innerDict.items():
            multilevel_dictionary[(outerKey, innerKey)] = values
    feature_file_name = results_path + feature + '_' + full_results_file_name
    pd.DataFrame(dict([(k, pd.Series(v)) for k, v in multilevel_dictionary.items()])).to_csv(feature_file_name, index=False)
    results.pop(feature, None)


def summarize_results(results_df, validation):
    # to delete for both cases: STD_CV and LOO
    columns_name_to_delete = ['cv', 'support_negative', 'support_positive']
    if validation == LOO:
        columns_name_to_delete += ['chemical-name', 'inchi']

    results_df.drop(columns_name_to_delete, axis=1, inplace=True)


def folder_for(parameters):
    folder = ''
    if parameters['model']['method'] == KNN:
        folder += 'knn'
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
    return folder + '/'
    # @TODO: Case GBC
