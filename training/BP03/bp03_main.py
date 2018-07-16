
import json
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
import training.BP03.settings
from training.func_use import create_monotone_string
from training.func_use import variable_importances
from training.func_use import error_metric
from training.func_use import build_data
from training.func_use import build_data_one
from training.func_use import data_feature_selected
from training.func_use import feature_select
from training.prepare import split_data
import general_settings
import itertools
from progressbar import ProgressBar
import time
import random
import os

def bp03_model(train,test,prj_info,TMP=1234):
    """Training pilot
    train -- pandas dataframe
    test -- pandas dataframe
    prj_info -- dictionnary containing projet information (response...)
    """
    #Naive tuning parameter
    print('    Naive tuning parameters')
    best_experiment,y_test,X_test,W_test,best_lightgbm,data_results_tp = lightgbm_tp(train,prj_info,training.BP03.settings.bp03_param)
    data_results_tp.to_csv(prj_info['OUTPUT_PATH'] + "tp_naive_" + str(TMP) + ".csv")
    #Feature selection
    print('    Feature selection')
    variables_selected,variable_data = feature_select(X_test, y_test,W_test, best_lightgbm, prj_info['METRIC'])
    variable_data.to_csv(prj_info['OUTPUT_PATH'] + "fs_" + str(TMP) + ".csv")
    train,test = data_feature_selected(train,test,variables_selected,list(prj_info['PRJ_COLUMN'].values()))
    #Deep tuning
    print('    Deep tuning parameters')
    best_experiment,y_test,X_test,W_test,best_lightgbm,data_results_tp = lightgbm_tp(train,prj_info,training.BP03.settings.bp03_param)
    data_results_tp.to_csv(prj_info['OUTPUT_PATH']+ "tp_deep_" + str(TMP) + ".csv")
    #Final Model
    print('    Final model')
    lightgbm,pred_fold,pred_test = lightgbm_final_model(train,test,prj_info,best_experiment)
    lightgbm.save_model(prj_info['OUTPUT_PATH']+ "model_final_" + str(TMP) + ".txt")
    #Save prediction
    pred_fold.to_csv(prj_info['OUTPUT_PATH']+ "pred_fold_" + str(TMP) + ".csv")
    pred_test.to_csv(prj_info['OUTPUT_PATH']+ "pred_test_" + str(TMP) + ".csv")

    return lightgbm,pred_fold,pred_test,variables_selected

def bp03_scoring(model,data,input_prj):
    """Scoring bp01
    model -- lightgbm model
    data -- pandas frame
    input_prj -- dictionnary containing projet information (response...)
    """
    #Prepare data
    y,X,W,O,monotonicity_vec = build_data_one(data,input_prj)
    #Score
    pred = model.predict(X, num_iteration=model.best_iteration)
    pred = pd.DataFrame(data={input_prj['PRJ_COLUMN']['INDEX']: data[input_prj['PRJ_COLUMN']['INDEX']], 'Pred' : pred})
    return pred

def lightgbm_tp(train,prj_info,setting):
    """Tuning parameters
    train -- pandas dataframe
    prj_info -- dictionnary containing projet information (response...)
    setting -- dictionnary containing settings
    """
    #Split data
    #train_,test_ = split_data(train,prj_info['PRJ_COLUMN']['FOLD_ASSIGN'])

    #Build data
    #y_train,y_test,X_train,X_test,W_train,W_test,O_train,monotonicity_vec = build_data(train_,test_,prj_info)

    #Dataset lightgbm
    #lgb_train = lgb.Dataset(X_train, y_train,weight=W_train,init_score = O_train)     
    #lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    
    #params
    #param_mono = {'monotone_constraints' : monotonicity_vec}
    params = setting['params'] 
    keys, values = zip(*params.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    #Loop
    data_results = pd.DataFrame([])
    timeout = time.time() + setting['naive_tp_time'] 
    loop_exp = 0
    exp_tested = []
    while True:
        #Break if time out
        if time.time() > timeout and data_results.shape[0]>2:
            break
        #Experiment
        exp = random.choice(experiments)
        #exp.update(param_mono)
        #Model
        metric_cv_test = []
        metric_cv_train = []
        for fold in range(1,max(train[prj_info['PRJ_COLUMN']['FOLD_ASSIGN']])+1):
            train_fold = train[train[prj_info['PRJ_COLUMN']['FOLD_ASSIGN']] != fold]
            test_fold = train[train[prj_info['PRJ_COLUMN']['FOLD_ASSIGN']] == fold]
            test_fold_idx = train[train[prj_info['PRJ_COLUMN']['FOLD_ASSIGN']] == fold][prj_info['PRJ_COLUMN']['INDEX']]
            y_train_fold,y_valid_fold,X_train_fold,X_valid_fold,W_train_fold,W_test_fold,O_train_fold,monotonicity_vec = build_data(train_fold,test_fold,prj_info)
                
            #Build datset
            lgb_train_fold = lgb.Dataset(X_train_fold, y_train_fold,weight=W_train_fold,init_score = O_train_fold)     
            lgb_eval_fold = lgb.Dataset(X_valid_fold, y_valid_fold, reference=lgb_train_fold)
            
            lightgbm = lgb.train(
                exp,
                train_set = lgb_train_fold,
                num_boost_round = 5000,
                early_stopping_rounds = 20,
                valid_sets=lgb_eval_fold,
                verbose_eval = False)
            #Predict
            pred_train_fold = lightgbm.predict(X_train_fold, num_iteration=lightgbm.best_iteration)
            pred_test_fold = lightgbm.predict(X_valid_fold, num_iteration=lightgbm.best_iteration)
            
            #Metric
            metric_train_fold = error_metric(y_train_fold,pred_train_fold,W_train_fold,prj_info['METRIC'])
            metric_test_fold = error_metric(y_valid_fold,pred_test_fold,W_test_fold,prj_info['METRIC'])
            metric_cv_test.append(metric_test_fold)
            metric_cv_train.append(metric_train_fold)

        metric_train = np.mean(metric_cv_train)
        metric_test = np.mean(metric_cv_test)
        if not os.path.exists(prj_info['OUTPUT_PATH']+ "shadow/"):
            os.makedirs(prj_info['OUTPUT_PATH']+ "shadow/")
        lightgbm.save_model(prj_info['OUTPUT_PATH']+ "shadow/" +str(loop_exp)+"model.txt")

        #Save results
        data_results_ = pd.DataFrame.from_dict(exp, orient='index')
        data_results_ = data_results_.transpose()
        data_results_["train"] = metric_train
        data_results_["test"] = metric_test
        data_results = data_results.append(data_results_)

        exp_tested.append(exp)
        loop_exp = loop_exp + 1
        
    #Find max experiment
    data_results = data_results.reset_index(drop=True)
    best_experiment_index = data_results["test"].idxmax()
    best_experiment = exp_tested[best_experiment_index]
    #Best model
    best_lightgbm = lgb.Booster(model_file=prj_info['OUTPUT_PATH']+  "shadow/" + str(best_experiment_index)+"model.txt")
    print("        " + str(loop_exp+1) + " models built")
    
    return best_experiment,y_valid_fold,X_valid_fold,W_test_fold,best_lightgbm,data_results

def lightgbm_final_model(train,test,prj_info,best_experiment):
    """Lightgbm final model
    train -- pandas dataframe
    test -- pandas dataframe
    prj_info -- dictionnary containing projet information (response...)
    best_experiment -- dictionnary containing settings
    """  
    #Param
    params = best_experiment
    #Build data
    y_train,y_test,X_train,X_test,W_train,W_test,O_train,monotonicity_vec = build_data(train,test,prj_info)
    
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
            
        #Build datset
        lgb_train_fold = lgb.Dataset(X_train_fold, y_train_fold,weight=W_train_fold,init_score = O_train)     
        lgb_eval_fold = lgb.Dataset(X_valid_fold, y_valid_fold, reference=lgb_train_fold)
        
        #Model cv
        lightgbm_cv = lgb.train(
            params,
            num_boost_round = 5000,
            early_stopping_rounds = 30,
            train_set = lgb_train_fold,
            valid_sets=lgb_eval_fold,
            verbose_eval = False)

        #Predict
        pred_valid_fold = lightgbm_cv.predict(X_valid_fold, num_iteration=lightgbm_cv.best_iteration)
        pred_fold_data = pd.DataFrame(data={prj_info['PRJ_COLUMN']['INDEX']: test_fold_idx, 'Pred' : pred_valid_fold})
        pred_fold = pred_fold.append(pred_fold_data)
        #Metric
        metric_test_cv = error_metric(y_valid_fold,pred_valid_fold,W_test_fold,prj_info['METRIC'])
        print('              Metric: ' + str(metric_test_cv))
        #Save results
        metric_cv.append(metric_test_cv)
        #Save best It
        best_it.append(lightgbm_cv.best_iteration)
        print('              Best iteration: ' + str(lightgbm_cv.best_iteration))
        
    metric_cv_mean = np.mean(metric_cv)
    best_it_mean = np.mean(best_it)
    #Full Model
    lgb_train = lgb.Dataset(X_train, y_train,weight=W_train,init_score = O_train)     

    print('         Full model')
    lightgbm = lgb.train(
        params,
        num_boost_round = int(round(best_it_mean)) +int(round(0.05*best_it_mean)),
        train_set = lgb_train,
        verbose_eval = False)

    pred_test = lightgbm.predict(X_test)
    metric_test = error_metric(y_test,pred_test,W_test,prj_info['METRIC'])
    pred_test = pd.DataFrame(data={prj_info['PRJ_COLUMN']['INDEX']: test[prj_info['PRJ_COLUMN']['INDEX']], 'Pred' : pred_test})
    
    print("    Fold mean " + prj_info['METRIC'] + " : " + str(metric_cv_mean))
    print("    Test " + prj_info['METRIC'] + " : " + str(metric_test))
    
    return lightgbm,pred_fold,pred_test
