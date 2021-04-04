from src.config import file_name_for
from src.config import results_path


def save_and_summary(results, parameters):
    full_results_file_name = file_name_for(parameters)[:-1] + '.csv'
    summary_results_file_name = 'summary_' + full_results_file_name

    results.to_csv(results_path + full_results_file_name, index=False)
    results.drop(['cv', 'support_negative', 'support_positive'], axis=1, inplace=True)

    std = results.groupby('dataset_index').std().add_suffix('_std')
    min_results = results.groupby('dataset_index').min().add_suffix('_min')
    max_results = results.groupby('dataset_index').max().add_suffix('_max')
    summary = results.groupby('dataset_index').mean().add_suffix('_mean')

    summary = summary.join(std).join(min_results).join(max_results)

    summary.to_csv(results_path + summary_results_file_name)


