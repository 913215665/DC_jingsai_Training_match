
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def mse_cv(model, data, target):
    print('模型评价...')
    kf = KFold(n_splits=5, shuffle=True, random_state=33).get_n_splits(data.values)  # 5等分
    crvs = cross_val_score(model, data.values, target, scoring='mean_squared_error', cv=kf)
    mse = abs(crvs)
    return mse


def predict_lasso(train, test):
    print('Lasso 模型预测中...')
    target = train.pop('pm2.5') / 1000

    # alpha 选0.5-0.8
    lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.5, random_state=111))
    score = mse_cv(lasso, train, target)
    print("Lasso score: {:.4f}({:.4f})\n".format(score.mean(), score.std()))

    lasso.fit(train, target)
    pred_lasso_y = lasso.predict(test)
    pred_lasso_y = pd.DataFrame(pred_lasso_y, index=None, columns=['pm2.5'])
    return pred_lasso_y


def predict_gboost(train, test):
    print('GBoost 模型预测中...')
    target = train.pop('pm2.5') / 1000

    GBoost = GradientBoostingRegressor(n_estimators=100,
                                       learning_rate=0.05,
                                       max_depth=20,
                                       max_features='sqrt',
                                       min_samples_leaf=15,
                                       min_samples_split=10,
                                       loss='huber',
                                       random_state=5
                                       )

    score = mse_cv(GBoost, train, target)
    print("GBoost score: {:.4f}({:.4f})\n".format(score.mean(), score.std()))

    GBoost.fit(train, target)
    pred_gboost_y = GBoost.predict(test) * 1000
    pred_gboost_y = pd.DataFrame(pred_gboost_y, index=None, columns=['pm2.5'])
    return pred_gboost_y


def predict_xgb(train, test):
    print('model_xgb 模型预测中...')
    target = train.pop('pm2.5') / 1000

    model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603,
                                 gamma=0.0468,
                                 learning_rate=0.05,
                                 max_depth=3,
                                 min_child_weight=1,
                                 n_estimators=2200,
                                 reg_alpha=0.4640,
                                 reg_lambda=0.8571,
                                 subsample=0.5213,
                                 random_state=7,
                                 nthread=-1
                                 )

    score = mse_cv(model_xgb, train, target)
    print("model_xgb score: {:.4f}({:.4f})\n".format(score.mean(), score.std()))

    model_xgb.fit(train, target)
    pred_xgb_y = model_xgb.predict(test) * 1000
    pred_xgb_y = pd.DataFrame(pred_xgb_y, index=None, columns=['pm2.5'])
    return pred_xgb_y


def predict_lgb(train, test):
    print('model_lgb 模型预测中...')
    target = train.pop('pm2.5') / 1000

    model_lgb = lgb.LGBMRegressor(objective='regression',
                                  num_leaves=5,
                                  learning_rate=0.05,
                                  n_estimators=200,
                                  max_bin=55,
                                  bagging_fraction=0.8,
                                  bagging_freq=5,
                                  feature_fraction=0.2319,
                                  feature_fraction_seed=9,
                                  bagging_seed=9,
                                  min_data_in_leaf=6, min_sum_hessian_in_leaf=11)

    score = mse_cv(model_lgb, train, target)
    print("model_lgb score: {:.4f}({:.4f})\n".format(score.mean(), score.std()))

    model_lgb.fit(train, target)
    pred_lgb_y = model_lgb.predict(test) * 1000
    pred_lgb_y = pd.DataFrame(pred_lgb_y, index=None, columns=['pm2.5'])
    return pred_lgb_y
