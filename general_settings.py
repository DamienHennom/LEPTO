########################################################
# SETTINGS
########################################################

PRJ_SET = {
    
    'INPUT_PATH_DATA' : 'E:/DATA_SCIENCE/USE_CASES/QUANTUM/DATA/final_data.csv',
    'OUTPUT_PATH' : 'E:/DATA_SCIENCE/USE_CASES/QUANTUM/OUTPUT/',
    'CODE_DIRECTORY' : 'E:/DATA_SCIENCE/USE_CASES/QUANTUM/PIPELINE/',
    
    'PRJ_COLUMN' : {
        'EXPOSURE' : None,
        'OFFSET'   : None,
        'WEIGHT'   : None,
        'INDEX'    : "id",
        'RESPONSE' : 'default_next_month',
        'FOLD_ASSIGN' : "FOLD", # Max fold use as private
    },

    'PROBLEM' : 'binary',
    'METRIC'  : 'auc',
    'FOLD'    : 5, #Not used
    'SEED'    : 1234,
    'POSITIVE_MONO' : [],
    'NEGATIVE_MONO' : []

    }


