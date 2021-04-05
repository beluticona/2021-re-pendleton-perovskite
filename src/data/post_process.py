from src.config import file_name_for
from src.config import results_path
import pandas as pd
import collections
from src import constants


def save_and_summary(results, parameters):
    full_results_file_name = file_name_for(parameters)[:-1] + '.csv'
    summary_results_file_name = 'summary_' + full_results_file_name

    if parameters['model']['method'] == constants.GBC:
        save_feature_importance(full_results_file_name, results, constants.FEAT_VALUES_IMPORTANCE)
        save_feature_importance(full_results_file_name, results, constants.FEAT_NAMES_IMPORTANCE)

    results_df = pd.DataFrame.from_dict(results, orient='columns')

    results_df.to_csv(results_path + full_results_file_name, index=False)
    results_df.drop(['cv', 'support_negative', 'support_positive'], axis=1, inplace=True)

    std = results_df.groupby('dataset_index').std().add_suffix('_std')
    min_results = results_df.groupby('dataset_index').min().add_suffix('_min')
    max_results = results_df.groupby('dataset_index').max().add_suffix('_max')
    summary = results_df.groupby('dataset_index').mean().add_suffix('_mean')

    summary = summary.join(std).join(min_results).join(max_results)

    summary.to_csv(results_path + summary_results_file_name)


def save_feature_importance(full_results_file_name, results, feature):
    pre_multilevel_dictionary = collections.defaultdict(dict)
    for i in range(len(results['dataset_index'])):
        pre_multilevel_dictionary[results['dataset_index'][i]][results['cv'][i]] = results[feature][i]
    multilevel_dictionary = {}
    for outerKey, innerDict in pre_multilevel_dictionary.items():
        for innerKey, values in innerDict.items():
            multilevel_dictionary[(outerKey, innerKey)] = values
    feature_file_name = results_path + feature + '_' + full_results_file_name
    pd.DataFrame(dict([(k, pd.Series(v)) for k, v in multilevel_dictionary.items()])).to_csv(feature_file_name, index=False)
    results.pop(feature, None)
