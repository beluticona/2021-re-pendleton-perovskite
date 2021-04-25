# config.py

from pathlib import Path

data_dir = Path('.')
data_path = 'data/raw/full-perovskite-data.csv' 
data_types_path = 'data/raw/types.json'
parameters_path = 'parameters.yaml'
results_path = 'results/alternative/cant_exp/'


def file_name_for(parameters):
    file_name = ''
    for setting in parameters:
        if isinstance(parameters[setting], dict):
            if setting in ['norm', 'std']: file_name += setting + '_'
            file_name += file_name_for(parameters[setting])
        elif not isinstance(parameters[setting], bool):
            file_name += setting + '_' + str(parameters[setting]).lower() + '_'
        elif parameters[setting]:
            file_name += setting + '_'
    return file_name


def set_results_folder(full_results_path):
    Path(full_results_path).mkdir(parents=True, exist_ok=True)

