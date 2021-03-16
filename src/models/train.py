
def get_solUD_model_columns(df_columns):
	solUD_model = list(filter(lambda column_name:column_name.startswith('_rxn_') or column_name.startswith('_feat_') and not column_name.startswith('_rxn_v0'), df_columns))
	solUD_model.remove('_rxn_organic-inchikey')
	return solUD_model

# Select columns names involved in each model
def filter_data_for_solV(df):
	solUD_model = get_solUD_model_columns(df.columns.to_list)

	solV_model = list(filter(lambda column_name: not column_name.startswith('_rxn_M_'), solUD_model))

	# Select data involved in each model
	solV_data = df[solV_model].reset_index(drop=True)
	
	return solV_data

def filter_data_for_solUD(df):
	solUD_model = get_solUD_model_columns(df.columns.to_list())
	
	solUD_data = df[solUD_model].reset_index(drop=True)
	
	return solUD_data

# columns names | new data columns

# col


def std_train_test(selected_data, model_parameters, crystal_score, inchikeys_count):
		if(model_parameters["cv_folds"] == 1):



