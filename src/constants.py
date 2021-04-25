# Classifiers
KNN = 1
GBC = 2
SVM = 3

# Methods
VOTING = 0
STD_CV = 1
LOO = 2

# Solution volume
SOLV_MODEL = 1
SOLUD_MODEL = 2
NO_MODEL = 0

FEAT_VALUES_IMPORTANCE = 'feats_importance_values'
FEAT_NAMES_IMPORTANCE = 'feats_names_by_importance'

# Dictionary from inchi key to its chemical name
INCHI_TO_CHEMNAME = {'null': 'null',
                     'YEJRWHAVMIAJKC-UHFFFAOYSA-N': 'Gamma-Butyrolactone',
                     'IAZDPXIOMUYVGZ-UHFFFAOYSA-N': 'Dimethyl sulfoxide',
                     'BDAGIHXWWSANSR-UHFFFAOYSA-N': 'Formic Acid',
                     'RQQRAHKHDFPBMC-UHFFFAOYSA-L': 'Lead Diiodide',
                     'XFYICZOIWSBQSK-UHFFFAOYSA-N': 'Ethylammonium Iodide',
                     'LLWRXQXPJMPHLR-UHFFFAOYSA-N': 'Methylammonium iodide',
                     'UPHCENSIMPJEIS-UHFFFAOYSA-N': 'Phenethylammonium iodide',
                     'GGYGJCFIYJVWIP-UHFFFAOYSA-N': 'Acetamidinium iodide',
                     'CALQKRVFTWDYDG-UHFFFAOYSA-N': 'n-Butylammonium iodide',
                     'UUDRLGYROXTISK-UHFFFAOYSA-N': 'Guanidinium iodide',
                     'YMWUJEATGCHHMB-UHFFFAOYSA-N': 'Dichloromethane',
                     'JMXLWMIFDJCGBV-UHFFFAOYSA-N': 'Dimethylammonium iodide',
                     'KFQARYBEAKAXIC-UHFFFAOYSA-N': 'Phenylammonium Iodide',
                     'NLJDBTZLVTWXRG-UHFFFAOYSA-N': 't-Butylammonium Iodide',
                     'GIAPQOZCVIEHNY-UHFFFAOYSA-N': 'N-propylammonium Iodide',
                     'QHJPGANWSLEMTI-UHFFFAOYSA-N': 'Formamidinium Iodide',
                     'WXTNTIQDYHIFEG-UHFFFAOYSA-N': '1,4-Diazabicyclo[2,2,2]octane-1,4-diium Iodide',
                     'LCTUISCIGMWMAT-UHFFFAOYSA-N': '4-Fluoro-Benzylammonium iodide',
                     'NOHLSFNWSBZSBW-UHFFFAOYSA-N': '4-Fluoro-Phenethylammonium iodide',
                     'FJFIJIDZQADKEE-UHFFFAOYSA-N': '4-Fluoro-Phenylammonium iodide',
                     'QRFXELVDJSDWHX-UHFFFAOYSA-N': '4-Methoxy-Phenylammonium iodide',
                     'SQXJHWOXNLTOOO-UHFFFAOYSA-N': '4-Trifluoromethyl-Benzylammonium iodide',
                     'KOAGKPNEVYEZDU-UHFFFAOYSA-N': '4-Trifluoromethyl-Phenylammonium iodide',
                     'MVPPADPHJFYWMZ-UHFFFAOYSA-N': 'chlorobenzene',
                     'CWJKVUQGXKYWTR-UHFFFAOYSA-N': 'Acetamidinium bromide',
                     'QJFMCHRSDOLMHA-UHFFFAOYSA-N': 'Benzylammonium Bromide',
                     'PPCHYMCMRUGLHR-UHFFFAOYSA-N': 'Benzylammonium Iodide',
                     'XAKAQFUGWUAPJN-UHFFFAOYSA-N': 'Beta Alanine Hydroiodide',
                     'KOECRLKKXSXCPB-UHFFFAOYSA-K': 'Bismuth iodide',
                     'XQPRBTXUXXVTKB-UHFFFAOYSA-M': 'Cesium iodide',
                     'ZMXDDKWLCZADIW-UHFFFAOYSA-N': 'Dimethylformamide',
                     'BCQZYUOYVLJOPE-UHFFFAOYSA-N': 'Ethane-1,2-diammonium bromide',
                     'IWNWLPUNKAYUAW-UHFFFAOYSA-N': 'Ethane-1,2-diammonium iodide',
                     'PNZDZRMOBIIQTC-UHFFFAOYSA-N': 'Ethylammonium bromide',
                     'QWANGZFTSGZRPZ-UHFFFAOYSA-N': 'Formamidinium bromide',
                     'VQNVZLDDLJBKNS-UHFFFAOYSA-N': 'Guanidinium bromide',
                     'VMLAEGAAHIIWJX-UHFFFAOYSA-N': 'i-Propylammonium iodide',
                     'JBOIAZWJIACNJF-UHFFFAOYSA-N': 'Imidazolium Iodide',
                     'RFYSBVUZWGEPBE-UHFFFAOYSA-N': 'iso-Butylammonium bromide',
                     'FCTHQYIDLRRROX-UHFFFAOYSA-N': 'iso-Butylammonium iodide',
                     'UZHWWTHDRVLCJU-UHFFFAOYSA-N': 'iso-Pentylammonium iodide',
                     'MCEUZMYFCCOOQO-UHFFFAOYSA-L': 'Lead(II) acetate trihydrate',
                     'ZASWJUOMEGBQCQ-UHFFFAOYSA-L': 'Lead(II) bromide',
                     'ISWNAMNOYHCTSB-UHFFFAOYSA-N': 'Methylammonium bromide',
                     'VAWHFUNJDMQUSB-UHFFFAOYSA-N': 'Morpholinium Iodide',
                     'VZXFEELLBDNLAL-UHFFFAOYSA-N': 'n-Dodecylammonium bromide',
                     'PXWSKGXEHZHFJA-UHFFFAOYSA-N': 'n-Dodecylammonium iodide',
                     'VNAAUNTYIONOHR-UHFFFAOYSA-N': 'n-Hexylammonium iodide',
                     'HBZSVMFYMAOGRS-UHFFFAOYSA-N': 'n-Octylammonium Iodide',
                     'FEUPHURYMJEUIH-UHFFFAOYSA-N': 'neo-Pentylammonium bromide',
                     'CQWGDVVCKBJLNX-UHFFFAOYSA-N': 'neo-Pentylammonium iodide',
                     'IRAGENYJMTVCCV-UHFFFAOYSA-N': 'Phenethylammonium bromide',
                     'UXWKNNJFYZFNDI-UHFFFAOYSA-N': 'piperazine dihydrobromide',
                     'QZCGFUVVXNFSLE-UHFFFAOYSA-N': 'Piperazine-1,4-diium iodide',
                     'HBPSMMXRESDUSG-UHFFFAOYSA-N': 'Piperidinium Iodide',
                     'IMROMDMJAWUWLK-UHFFFAOYSA-N': 'Poly(vinyl alcohol), Mw89000-98000, >99% hydrolyzed)',
                     'QNBVYCDYFJUNLO-UHDJGPCESA-N': 'Pralidoxime iodide',
                     'UMDDLGMCNFAZDX-UHFFFAOYSA-O': 'Propane-1,3-diammonium iodide',
                     'VFDOIPKMSSDMCV-UHFFFAOYSA-N': 'Pyrrolidinium Bromide',
                     'DMFMZFFIQRMJQZ-UHFFFAOYSA-N': 'Pyrrolidinium Iodide',
                     'DYEHDACATJUKSZ-UHFFFAOYSA-N': 'Quinuclidin-1-ium bromide',
                     'LYHPZBKXSHVBDW-UHFFFAOYSA-N': 'Quinuclidin-1-ium iodide',
                     'UXYJHTKQEFCXBJ-UHFFFAOYSA-N': 'tert-Octylammonium iodide',
                     'BJDYCCHRZIFCGN-UHFFFAOYSA-N': 'Pyridinium Iodide',
                     'ZEVRFFCPALTVDN-UHFFFAOYSA-N': 'Cyclohexylmethylammonium iodide',
                     'WGYRINYTHSORGH-UHFFFAOYSA-N': 'Cyclohexylammonium iodide',
                     'XZUCBFLUEBDNSJ-UHFFFAOYSA-N': 'Butane-1,4-diammonium Iodide',
                     'RYYSZNVPBLKLRS-UHFFFAOYSA-N': '1,4-Benzene diammonium iodide',
                     'DWOWCUCDJIERQX-UHFFFAOYSA-M': '5-Azaspiro[4.4]nonan-5-ium iodide',
                     'YYMLRIWBISZOMT-UHFFFAOYSA-N': 'Diethylammonium iodide',
                     'UVLZLKCGKYLKOR-UHFFFAOYSA-N': '2-Pyrrolidin-1-ium-1-ylethylammonium iodide',
                     'BAMDIFIROXTEEM-UHFFFAOYSA-N': 'N,N-Dimethylethane- 1,2-diammonium iodide',
                     'JERSPYRKVMAEJY-UHFFFAOYSA-N': 'N,N-dimethylpropane- 1,3-diammonium iodide',
                     'NXRUEVJQMBGVAT-UHFFFAOYSA-N': 'N,N-Diethylpropane-1,3-diammonium iodide',
                     'N/A': 'None'}
