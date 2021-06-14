# config.py

from pathlib import Path

# Paths to used files
data_dir = Path('./')
raw_data_path = data_dir / 'data/raw/full-perovskite-data.csv'
training_dataset_path = data_dir / 'data/training-perovskite-data.csv'
testing_dataset_path = data_dir / 'data/testing-perovskite-data.csv'
data_types_path = data_dir / 'data/raw/types.json'
parameters_path = data_dir / 'parameters.yaml'
notebook_path = data_dir /'notebooks/results/'
chemical_inventory_path = data_dir / 'data/metadata/chemical_inventory.csv'

# Result path
results_path = data_dir / 'results/alternative/importance/10'

# Generate filename for results according to parameters
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

