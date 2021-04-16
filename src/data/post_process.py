from src.config import file_name_for
from src.config import results_path
from src.config import set_results_folder
import pandas as pd
import collections
from src import constants


def save_results(full_results, parameters):
    full_results_descriptions = file_name_for(parameters)[:-1] + '.csv'
    full_results_path = results_path + folder_for(parameters)
    set_results_folder(full_results_path)

    if parameters['model']['method'] == constants.GBC:
        # TO REFACTOR
        save_feature_importance(full_results_descriptions, full_results, constants.FEAT_VALUES_IMPORTANCE)
        save_feature_importance(full_results_descriptions, full_results, constants.FEAT_NAMES_IMPORTANCE)
    if parameters['intrpl']:
        save_full_and_summary_for(full_results['std'], full_results_path, 'StandardTestTrain_Full.csv', 'StandardTestTrain_Summary.csv', parameters)
    if parameters['exptrpl']:
        save_full_and_summary_for(full_results['loo'], full_results_path, 'LeaveOneOut_Full.csv', 'LeaveOneOut_Summary.csv', parameters)

    f = open(results_path + 'info.txt', "a")
    f.write(full_results_descriptions)
    f.close()


def save_full_and_summary_for(dict_results, full_results_path, full_result_filename, summary_filename, parameters):
    df_results = pd.DataFrame.from_dict(dict_results, orient='columns')
    df_results.to_csv(full_results_path + full_result_filename)
    filter_only_metrics_columns(parameters, df_results)

    std = df_results.groupby('data_index').std().add_suffix('_std')
    min_results = df_results.groupby('data_index').min().add_suffix('_min')
    max_results = df_results.groupby('data_index').max().add_suffix('_max')
    summary = df_results.groupby('data_index').mean().add_suffix('_mean')

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


def filter_only_metrics_columns(parameters, results_df):
    if parameters['extrpl']:
        results_df.drop(['cv', 'chemical-name', 'inchi', 'support_negative', 'support_positive'], axis=1, inplace=True)
    else:
        results_df.drop(['cv', 'support_negative', 'support_positive'], axis=1, inplace=True)


def folder_for(parameters):
    folder = ''
    if parameters['model']['method'] == constants.KNN:
        folder += 'knn'
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
