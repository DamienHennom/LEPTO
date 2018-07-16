
def prepare_dataset(dataset,prj_info):
    """Prepare dataset
    dataset -- pandas dataset
    prj_info -- dictionnary containing projet information (response...)
    """
    #Split data
    train,test = split_data(dataset,prj_info['FOLD_ASSIGN'])
    #Handle exposure
    train = exposure_handle(train,prj_info['RESPONSE'],prj_info['EXPOSURE'],prj_info['WEIGHT'])
    return train,test

def split_data(dataset,FOLD_ASSIGN):
    """Split dataset
    dataset -- pandas dataset
    FOLD_ASSIGN -- string (column for fold assignement)
    """
    train = dataset[dataset[FOLD_ASSIGN] < max(dataset[FOLD_ASSIGN])]
    test = dataset[dataset[FOLD_ASSIGN] == max(dataset[FOLD_ASSIGN])]
    return train,test

def exposure_handle(train,RESPONSE,EXPOSURE,WEIGHT):
    """Prepare for exposure
    train -- pandas dataset
    RESPONSE -- string (column response)
    EXPOSURE -- string (column exposure)
    WEIGHT -- string (column weight)
    """
    #Copy data
    train = train.copy()
    #Handle exposure
    if EXPOSURE != None:
        train[RESPONSE] =  train[RESPONSE]/train[EXPOSURE]
        if WEIGHT != None:
            train[WEIGHT] = train[WEIGHT]*train[EXPOSURE]
        
    return train
