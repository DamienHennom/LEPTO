
import general_settings

bp03_param = {
    #Naive tuning time sec
    'naive_tp_time' : 1800,
    #Deep tuning time sec
    'deep_tp_time' : 1800,
    #Param Lightgbm
    'params' : {
    'task'             : ['train'],
    'application'      : [general_settings.PRJ_SET['PROBLEM']],
    'metric'           : ['binary_logloss'],
    'tree_learner'     : ['serial'],
    'boosting_type'    : ['gbdt'],
    'num_threads'      : [11],
    'device'           : ['cpu'],
    #'num_boost_round'  : [500],
    #'early_stopping_rounds' : [10],
    
    'learning_rate'    : [0.005],
    'num_leaves'       : [60,70,80,90,100,150],
    'max_depth'        : [6,7,8,9,10],
    'min_data_in_leaf' : [10,30,50,80,100,150],
    'min_sum_hessian_in_leaf' : [1e-4],
    'feature_fraction' : [0.1,0.3,0.5,0.7,0.9],
    'feature_fraction_seed' : [general_settings.PRJ_SET['SEED']],
    'bagging_fraction' : [0.1,0.3,0.5,0.7,0.9],
    'bagging_seed '    : [general_settings.PRJ_SET['SEED']],
    'bagging_freq'     : [0,2,5,10,20,30,50],
    'verbose'          : [-1],
    'lambda_l1'        : [0,0.01,0.05,0.02],
    'lambda_l2'        : [0,0.01,0.05,0.02],
    'max_delta_step'   : [0,0.01,0.005,0.02],
    'min_split_gain'   : [0,0.01,0.005,0.02],
    'drop_rate'        : [0.1],
    'drop_seed'        : [general_settings.PRJ_SET['SEED']],
    'skip_drop'        : [0.5],
    'max_drop'         : [50],
    'uniform_drop'     : [False],
    'xgboost_dart_mode': [False],
    'top_rate'         : [0.2],
    'other_rate'       : [0.1],
    'min_data_per_group': [50],
    'max_cat_threshold' : [32],
    'cat_smooth'        : [10],
    'cat_l2'            : [10],
    'max_cat_to_onehot' : [4],
    'top_k'             : [20],

    'data_random_seed'  : [general_settings.PRJ_SET['SEED']],
    'sigmoid'           : [1.0],
    'alpha'             : [0.9],
    'fair_c'            : [1.0],
    'poisson_max_delta_step' : [0.7],
    'scale_pos_weight'  : [1.0],
    'boost_from_average': [True],
    'is_unbalance'      : [False,True],
    'max_position'      : [20],
    'label_gain'        : ['0,1,3,7,15,31,63,127,255,511,1023,2047,4095,8191,16383'],
    'num_class'         : [1],
    'reg_sqrt'          : [False],
    'tweedie_variance_power' : [1.5]
    }
    }

