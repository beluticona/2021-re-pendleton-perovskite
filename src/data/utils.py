

experiment_version = 1.1

GBL_inchikey = 'YEJRWHAVMIAJKC-UHFFFAOYSA-N'

dimethyl_ammonium_inchikey = 'MXLWMIFDJCGBV-UHFFFAOYSA-N'

def make_dataset(df):
	
	# Select data from version 1.1
	df.query('_raw_ExpVer == @experiment_version', inplace = True)

	# Select reactions where only GBL is used as solvent 
	df.query('_raw_reagent_0_chemicals_0_InChIKey == @GBL_inchikey', inplace = True)
	
	# Remove some anomalous entries with dimethyl ammonium still listed as the organic.	
	df.query('_raw_reagent_0_chemicals_0_InChIKey != @dimethyl_ammonium_inchikey', inplace = True) 
	
	## DECISION FUERTE, POR QUÉ NO CONSIDERAR LOS QUE TENGAN 3?
	# Collect inchikeys of solvents that had at least a succesful crystal
	
	succesful_inchikeys = df.query('_out_crystalscore == 4')['_rxn_organic-inchikey'].unique()
	
	succesful_reactions = (df[df['_rxn_organic-inchikey'].isin(succesful_inchikeys)])
	
	return succesful_reactions


	
	
	
	

	
	
	
	
	
