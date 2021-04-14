from src.config import file_name_for
from src.config import results_path
from src.config import set_results_folder
import pandas as pd
import collections
from src import constants


def save_and_summary(results, parameters):
    full_results_file_name = file_name_for(parameters)[:-1] + '.csv'
    # summary_results_file_name = 'summary_' + full_results_file_name
    full_results_path = results_path + folder_for(parameters)
    set_results_folder(full_results_path)

    if parameters['model']['method'] == constants.GBC:
        save_feature_importance(full_results_file_name, results, constants.FEAT_VALUES_IMPORTANCE)
        save_feature_importance(full_results_file_name, results, constants.FEAT_NAMES_IMPORTANCE)

    results_df = pd.DataFrame.from_dict(results, orient='columns')

    # results_df.to_csv(results_path + full_results_file_name, index=False)
    if parameters['intrpl']:
        results_df.to_csv(full_results_path + 'StandardTestTrain_Full.csv')
    else:
        results_df.to_csv(full_results_path + 'LeaveOneOut_Full.csv')

    filter_only_metrics_columns(parameters, results_df)

    std = results_df.groupby('data_index').std().add_suffix('_std')
    min_results = results_df.groupby('data_index').min().add_suffix('_min')
    max_results = results_df.groupby('data_index').max().add_suffix('_max')
    summary = results_df.groupby('data_index').mean().add_suffix('_mean')

    summary = summary.join(std).join(min_results).join(max_results)

    # summary.to_csv(results_path + summary_results_file_name)

    if parameters['intrpl']:
        results_df.to_csv(full_results_path + 'StandardTestTrain_Summary.csv')
    else:
        results_df.to_csv(full_results_path + 'LeaveOneOut_Summary.csv')

    f = open(results_path + 'info.txt', "a")
    f.write(full_results_file_name)
    f.close()


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
    if parameters['mode']['one-hot-encodinng']:
        folder += '_hot_encode'
    return folder + '/'
    # @TODO: Case GBC
