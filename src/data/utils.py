import pandas as pd
import numpy as np
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
    # @TODO: _feat_vanderwalls_volume no filtra nada, solV ya lo incluye por ser _featVanderWaalsVolme
    regex_for_chem_columns = ['_raw_reagent_.*_chemicals_.*_actual_amount$',
                              '_raw_*molweight',
                              '_feat_vanderwalls_volume',
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


def stratify(data, crystal_score, inchis_values):
    """
    Select 96 rows for each inch key.
    """
    stratified_data = pd.DataFrame()
    stratified_crystal_score = pd.DataFrame()
    inchis = np.unique(inchis_values)

    for i, inchi in enumerate(inchis):
        inchi_mask = inchis_values == inchi
        # bools de quienes son x
        total_amine = data[inchi_mask].reset_index(drop=True)
        amine_out = crystal_score[inchi_mask].reset_index(drop=True)

        # this is still experimental and can easily be changed.
        try:
            uniform_samples = np.random.choice(total_amine.index, size=96, replace=False)
        except Exception:
            uniform_samples = np.random.choice(total_amine.index, size=96)

        sampled_amine = total_amine.loc[uniform_samples]
        sampled_crystal_score = amine_out.loc[uniform_samples]

        # save pointer to where this amine lives in the stratified dataset.
        # this isn't needed for random-TTS, but makes doing the Leave-One-Amine-Out
        # train-test-splitting VERY EASY.
        # indicies[x] = np.array(range(96)) + i*96

        stratified_data = pd.concat([stratified_data, sampled_amine]).reset_index(drop=True)
        stratified_crystal_score = pd.concat([stratified_crystal_score, sampled_crystal_score]).reset_index(drop=True)

    stratified_crystal_score = stratified_crystal_score.iloc[:, 0]

    return stratified_data,  stratified_crystal_score

