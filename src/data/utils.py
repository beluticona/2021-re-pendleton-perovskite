import re


def get_sol_ud_model_columns(df_columns):
    sol_ud_model_columns = list(filter(lambda column_name: column_name.startswith('_rxn_') or column_name.startswith(
        '_feat_') and not column_name.startswith('_rxn_v0'), df_columns))
    sol_ud_model_columns.remove('_rxn_organic-inchikey')
    return sol_ud_model_columns


def get_sol_v_model_columns(df_columns):
    sol_ud_model_columns = get_sol_ud_model_columns(df_columns)
    sol_v_model_columns = list(filter(lambda column_name: not column_name.startswith('_rxn_M_'), sol_ud_model_columns))
    return sol_v_model_columns


def extend_by_regexes(df_columns, regexes, subset_columns_to_extend):
    for reg_string in regexes:
        reg = re.compile(reg_string)
        subset_columns_to_extend.extend(list(filter(reg.match, df_columns)))


def extend_with_chem_columns(df_columns, subset_columns_to_extend):
    regex_for_chem_columns = ['_raw_reagent_.*_chemicals_.*_actual_amount$',
                              '_raw_*molweight',
                              '_feat_VanderWaalsVolume',
                              '_raw_reagent_\d_volume']
    extend_by_regexes(df_columns, regex_for_chem_columns, subset_columns_to_extend)


def extend_with_rxn_columns(df_columns, subset_columns_to_extend, sol_ud_enable=False):
    rxn_regexes = lambda v1: [f"_raw{'_v1-' if v1 else '_'}M_.*_final",
                              f"_raw_reagent_\d{'_v1-' if v1 else '_'}conc_.*",
                              "_raw_reagent_\d_volume"]
    regex_for_rxn_columns = rxn_regexes(sol_ud_enable)
    extend_by_regexes(df_columns, regex_for_rxn_columns, subset_columns_to_extend)


def extend_with_reag_columns(df_columns, subset_columns_to_extend, sol_ud_enable=False):
    rxn_regexes = lambda v1: ['_raw_reagent_\d_volume$',
                              f"_raw_reagent_\d{'_v1-' if v1 else '_'}conc.*"]
    regex_for_rxn_columns = rxn_regexes(sol_ud_enable)
    extend_by_regexes(df_columns, regex_for_rxn_columns, subset_columns_to_extend)


def filter_required_data(df, type_sol_volume, chem_extend_enabled, exp_extend_enabled, reag_extend_enabled):
    df_columns = df.columns.to_list()
    sol_model_columns = []
    # @TODO Replace constant numbers for global constants
    if type_sol_volume == 1:
        sol_model_columns = get_sol_v_model_columns(df_columns)
    elif type_sol_volume == 2:
        sol_model_columns = get_sol_ud_model_columns(df_columns)

    if chem_extend_enabled:
        extend_with_chem_columns(df_columns, sol_model_columns)
        # @TODO: verify if this column can be deleted at the first preliminary filter
        # Clean reagent_5_chemical because it's full of zeros and null9
        df['_raw_reagent_5_chemicals_2_actual_amount'] = [0] * df.shape[0]
    elif exp_extend_enabled:
        extend_with_rxn_columns(df_columns, sol_model_columns, type_sol_volume == 2)
    elif reag_extend_enabled:
        extend_with_reag_columns(df_columns, sol_model_columns, type_sol_volume == 2)
    return df[sol_model_columns].fillna(0).reset_index(drop=True)
