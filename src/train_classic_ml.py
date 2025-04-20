import lightgbm as lgb
import catboost as cb


def train_catboost(X_train, y_train, X_val=None, y_val=None, plot=False, kwargs={}):   
    '''Train catboost regressor.'''
    pool_tr = cb.Pool(X_train, y_train)
    if X_val is not None:
        pool_eval = cb.Pool(X_val, y_val)
    else:
        pool_eval=None


    params = {
        'iterations': None,
        'depth': None, 
        'learning_rate': None, 
        'random_state': 1,
        'eval_metric': 'RMSE',
        'loss_function': 'RMSE',
        'task_type': 'CPU',
    }
    if kwargs:
        params.update(kwargs)
    model = cb.CatBoost(params)
    model.fit(
            pool_tr, 
            eval_set=pool_eval,
            use_best_model=True,
            verbose=10,
            early_stopping_rounds=100,
        )
    return model


def train_lgb(X_train, y_train, X_val=None, y_val=None, plot=False, kwargs={}):

    lgb_tr = lgb.Dataset(X_train, y_train)
    if X_val is not None:
        lgb_val = lgb.Dataset(X_val, y_val)
    else:
        lgb_val = None

    params = {
        'objective': 'regression',  # or custom callable
        'eta': 0.03,
        # 'lambda': 1e-3,

        'seed': 911,
        'num_threads': 32,
        'verbosity': 2,
        'metrics': 'rmse',
        'num_boost_round': 454
    }
    if kwargs:
        params.update(kwargs)
    model = lgb.train(
        params, lgb_tr  # basic
        #valid_sets=[lgb_val], valid_names=['валидация'],
        # feval=my_custom_metrics, # можно задать кастомные метрики для early_stopping
        # callbacks=[
        #     lgb.early_stopping(stopping_rounds=100, min_delta=0.),
        #     lgb.log_evaluation(period=10)  # чтобы выводились результаты подсчета метрики для early_stopping
        # ]
    )
    return model
