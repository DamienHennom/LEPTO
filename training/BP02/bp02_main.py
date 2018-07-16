
import pandas as pd
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
import sys
from training.BP02.settings import bp02_param
from training.prepare import split_data
from training.func_use import build_data
from sklearn import preprocessing
from training.func_use import error_metric
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Activation
from keras.models import Model
import pickle
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras.callbacks import History
from transform.transformation import GaussRankScaler
from keras import optimizers
from sklearn.metrics import roc_auc_score

def bp02_model(train,test,prj_info,TMP=1234):
    """Training pilot
    train -- pandas dataframe
    test -- pandas dataframe
    prj_info -- dictionnary containing projet information (response...)
    """
    #Check
    bp02_check(prj_info)
    #Autoencoder
    print('    Auto encoder')
    encoded_train,encoded_test = bp02_autoencoder(train,test,prj_info)
    #Final Model
    print('    Final model')
    keras_model,pred_fold,pred_test,variables_selected,le,scale = keras_final_model(encoded_train,encoded_test,prj_info,bp02_param)
    #keras_model,pred_fold,pred_test,variables_selected,le,scale = keras_final_model(train,test,prj_info,bp02_param)
    #Save Prediction
    pred_fold.to_csv(prj_info['OUTPUT_PATH']+ "pred_fold_" + str(TMP) + ".csv")
    pred_test.to_csv(prj_info['OUTPUT_PATH']+ "pred_test_" + str(TMP) + ".csv")
    #Save data transform
    pickle.dump(le, open(prj_info['OUTPUT_PATH']+ "encoder_" + str(TMP) + ".p", "wb" ))
    pickle.dump(scale, open(prj_info['OUTPUT_PATH']+ "scale_" + str(TMP) + ".p", "wb" ))
    
    return keras_model,pred_fold,pred_test,variables_selected

def bp02_autoencoder(train,test,prj_info,encoding_dim = 100):
    """Autoencoder
    train -- pandas dataframe
    test -- pandas dataframe
    prj_info -- dictionnary containing projet information (response...)
    encoding_dim -- int
    """
    #Copy data
    train_ = train.copy()
    test_ = test.copy()
    
    #Build data
    y_train,y_test,X_train,X_test,W_train,W_test,O_train,monotonicity_vec = build_data(train_,test_,prj_info)
    #Prep data
    X_train,le_X,scale_X = keras_prep_data(X_train)
    X_test,le_X,scale_X = keras_prep_data(X_test,le_X,scale_X)
    
    #Test set
    train_fold = train_[train_[prj_info['PRJ_COLUMN']['FOLD_ASSIGN']] < max(train_[prj_info['PRJ_COLUMN']['FOLD_ASSIGN']])]
    test_fold = train_[train_[prj_info['PRJ_COLUMN']['FOLD_ASSIGN']] == max(train_[prj_info['PRJ_COLUMN']['FOLD_ASSIGN']])]
    y_train_fold,y_valid_fold,X_train_fold,X_valid_fold,W_train_fold,W_test_fold,O_train_fold,monotonicity_vec = build_data(train_fold,test_fold,prj_info)
    #Prep data
    X_train_fold,le_X,scale_X = keras_prep_data(X_train_fold,le_X,scale_X)
    X_valid_fold,le_X,scale_X = keras_prep_data(X_valid_fold,le_X,scale_X)
    
    #Early stop
    early_stop = EarlyStopping(monitor='val_loss', patience=20, mode='auto')
    #Encoder
    input_dim = Input(shape=(X_train.shape[1],))

    encoded1 = Dense(30, activation = 'relu')(input_dim)#200
    encoded3 = Dense(50, activation = 'relu')(encoded1) #500
    encoded4 = Dense(encoding_dim, activation = 'relu')(encoded3)

    decoded1 = Dense(50, activation = 'relu')(encoded4)
    decoded3 = Dense(30, activation = 'relu')(decoded1)
    decoded4 = Dense(X_train.shape[1], activation = 'sigmoid')(decoded3)

    # Whole model
    autoencoder = Model(input_dim, decoded4)
    # Encoder model
    encoder = Model(input_dim, encoded4)

    #Compile Fit
    adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)
    autoencoder.compile(optimizer=adam, loss='mean_squared_error')
    autoencoder.fit(X_train_fold,X_train_fold,
                    epochs=1500,
                    batch_size=20,
                    shuffle=True,
                    verbose=2,
                    validation_data=(X_valid_fold, X_valid_fold),
                    callbacks=[early_stop])
    
    encoded_train = encoder.predict(X_train)
    encoded_test = encoder.predict(X_test)

    #Rebuild train test
    column = list(range(0,encoding_dim))
    column = [str(s) for s in column]
    column = ['autoencoder_' + s for s in column]
    train_new = pd.DataFrame(encoded_train, columns=column)
    test_new = pd.DataFrame(encoded_test, columns=column)

    train_new[prj_info['PRJ_COLUMN']['RESPONSE']] = y_train
    test_new[prj_info['PRJ_COLUMN']['RESPONSE']] = y_test
    train_new[prj_info['PRJ_COLUMN']['FOLD_ASSIGN']] = train_[prj_info['PRJ_COLUMN']['FOLD_ASSIGN']].values
    test_new[prj_info['PRJ_COLUMN']['FOLD_ASSIGN']] = test_[prj_info['PRJ_COLUMN']['FOLD_ASSIGN']].values
    train_new[prj_info['PRJ_COLUMN']['INDEX']] = train_[prj_info['PRJ_COLUMN']['INDEX']].values
    test_new[prj_info['PRJ_COLUMN']['INDEX']] = test_[prj_info['PRJ_COLUMN']['INDEX']].values
    
    return train_new, test_new

def bp02_check(prj_info) :
    """Check 
    prj_info -- dictionnary containing projet information (response...)
    """
    if prj_info['PRJ_COLUMN']['EXPOSURE'] != None :
        sys.exit("Exposure not supported for bp02")
    if prj_info['PRJ_COLUMN']['WEIGHT'] != None :
        sys.exit("Weight not supported for bp02")
    return None

def keras_prep_data(data,le = None,scale = None):
    """Prepare data for Keras model
    data -- pandas dataframe
    le -- preprocessing
    scale -- preprocessing
    """
    #Copy data
    data = data.copy()
    #Column
    categorical_feature = data.select_dtypes(include=['object']).columns.values
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numerical_feature = data.select_dtypes(include=numerics).columns.values
    #Fill NA
    data = data.fillna(data.mean())
    #Scale
    sc = GaussRankScaler()
    for feature in numerical_feature:
           data[feature] = sc.fit_transform(data[feature])
    #Encoder
    if le == None:
        le = preprocessing.OneHotEncoder()
        for feature in categorical_feature:
            data[feature] = le.fit_transform(data[feature])
    else:
        for feature in categorical_feature:
            data[feature] = le.transform(data[feature])
    #Scale predictors
    scale = preprocessing.StandardScaler()
    #if scale == None:
        #scale = preprocessing.StandardScaler()
        #data = scale.fit_transform(data)
    #else:
        #data = scale.transform(data)    
    return data,le,scale

def keras_prep_response(y,le = None):
    """Prepare y for Keras model
    y -- pandas series
    le -- preprocessing
    scale -- preprocessing
    """
    #copy y
    y = y.copy()
    #Encoder
    if le == None:
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)
    else:
        y = le.transform(y)
    return y,le

def keras_final_model(train,test,prj_info,settings):
    """Training pilot
    train -- pandas dataframe
    test -- pandas dataframe
    prj_info -- dictionnary containing projet information (response...)
    settings -- dictionnary containing settings
    """
    #Build data
    y_train,y_test,X_train,X_test,W_train,W_test,O_train,monotonicity_vec = build_data(train,test,prj_info)
    variables_selected = X_train.columns.values
    
    #Prep data
    X_train,le_X,scale_X = keras_prep_data(X_train)
    X_test,le_X,scale_X = keras_prep_data(X_test,le_X,scale_X)

    #Early stop
    early_stop = EarlyStopping(monitor='val_loss', patience=20, mode='auto') 
    
    #Model
    def bp02_model(input_dim = None):
        adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)
        model = Sequential()
        model.add(Dense(50, input_dim = input_dim, kernel_initializer='normal', activation='relu',kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dropout(0.1)) #1500
        model.add(Dense(20, kernel_initializer='normal', activation='relu',kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dropout(0.1)) #750
        model.add(Dense(20, kernel_initializer='normal', activation='relu',kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dropout(0.1))#750
        model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=adam)
        return model

    #Cv Model
    metric_cv = []
    pred_fold = pd.DataFrame([])
    best_it = []
    for fold in range(1,max(train[prj_info['PRJ_COLUMN']['FOLD_ASSIGN']])+1):
        print('         Fold ' + str(fold))
        train_fold = train[train[prj_info['PRJ_COLUMN']['FOLD_ASSIGN']] != fold]
        test_fold = train[train[prj_info['PRJ_COLUMN']['FOLD_ASSIGN']] == fold]
        test_fold_idx = train[train[prj_info['PRJ_COLUMN']['FOLD_ASSIGN']] == fold][prj_info['PRJ_COLUMN']['INDEX']]
        y_train_fold,y_valid_fold,X_train_fold,X_valid_fold,W_train_fold,W_test_fold,O_train_fold,monotonicity_vec = build_data(train_fold,test_fold,prj_info)

        #Prep data
        X_train_fold,le_X,scale_X = keras_prep_data(X_train_fold,le_X,scale_X)
        X_valid_fold,le_X,scale_X = keras_prep_data(X_valid_fold,le_X,scale_X)
        
        #Estimator
        clf_fold = KerasClassifier(build_fn = bp02_model,
                                   input_dim = X_train.shape[1],
                                   epochs = settings['params']['epochs'],
                                   batch_size = settings['params']['batch_size'],
                                   verbose = settings['params']['verbose'],
                                   callbacks=[early_stop])
        
        #Model cv
        history_fold =  clf_fold.fit(X_train_fold,y_train_fold, validation_data = (X_valid_fold,y_valid_fold))
        #Predict
        pred_valid_fold = clf_fold.predict_proba(X_valid_fold)
        pred_valid_fold = [item[1] for item in pred_valid_fold]
        pred_fold_data = pd.DataFrame(data={prj_info['PRJ_COLUMN']['INDEX']: test_fold_idx, 'Pred' : pred_valid_fold})
        pred_fold = pred_fold.append(pred_fold_data)
        #Metric
        metric_test_cv = error_metric(y_valid_fold,pred_valid_fold,W_test_fold,prj_info['METRIC'])
        print(metric_test_cv)
        #Save results
        metric_cv.append(metric_test_cv)
        #Save best Iteration
        best_it_fold = history_fold.history['val_loss'].index(min(history_fold.history['val_loss']))+1
        best_it.append(best_it_fold)

    metric_cv_mean = np.mean(metric_cv)
    best_it_mean = np.mean(best_it)
    
    #Full model
    print('         Full model')

    #Estimator
    clf = KerasClassifier(build_fn = bp02_model,
                          input_dim = X_train.shape[1],
                          epochs = int(round(best_it_mean)),
                          batch_size = settings['params']['batch_size'],
                          verbose = settings['params']['verbose'])
    
    clf.fit(X_train,y_train)

    pred_test = clf.predict_proba(X_test)
    pred_test = [item[1] for item in pred_test]
    metric_test = error_metric(y_test,pred_test,W_test,prj_info['METRIC'])
    pred_test = pd.DataFrame(data={prj_info['PRJ_COLUMN']['INDEX']: test[prj_info['PRJ_COLUMN']['INDEX']], 'Pred' : pred_test})
    
    print("    Fold mean " + prj_info['METRIC'] + " : " + str(metric_cv_mean))
    print("    Test " + prj_info['METRIC'] + " : " + str(metric_test))
    
    return clf,pred_fold,pred_test,variables_selected,le_X,scale_X
