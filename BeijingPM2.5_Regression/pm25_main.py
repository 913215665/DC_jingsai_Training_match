# coding:utf-8

"""

"""

import time
from pm25_data_import import read_data
from pm25_data_preprocessing import preprocessing
from pm25_data_prediction import predict
from pm25_data_prediction2 import predict_lasso, predict_gboost, predict_xgb, predict_lgb
import warnings
warnings.filterwarnings('ignore')


def main():
    # 读取数据
    test = read_data('pm25_test.csv')
    train = read_data('pm25_train.csv')

    # 数据预处理
    train_data, test_data = preprocessing(train, test)
    print(f'data shape:\ntrain--{train_data.shape}\ntest--{test_data.shape}')

    # 预测
    pred_y = predict(train_data, test_data, is_shuffle=True)
    pred_y.to_csv('./pm25_pred_1010.csv', index=False, header=['pm2.5'])

    # Lasso 模型预测
    # pred_lasso_y = predict_lasso(train_data, test_data)
    # pred_lasso_y.to_csv('./pm25_pred_lasso_y_1009.csv', index=False)

    # gboost 模型预测
    # pred_gboost_y = predict_gboost(train_data, test_data)
    # pred_gboost_y.to_csv('./pm25_pred_gboost_y_1009.csv', index=False)

    # xgb 模型预测
    # pred_xgb_y =  predict_xgb(train_data, test_data)
    # pred_xgb_y.to_csv('./pm25_pred_xgb_y_1009.csv', index=False)

    # lgb 模型预测
    # pred_lgb_y = predict_lgb(train_data, test_data)
    # pred_lgb_y.to_csv('./pm25_pred_lgb_y_1009.csv', index=False)


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()

    print("耗时：", end_time - start_time)