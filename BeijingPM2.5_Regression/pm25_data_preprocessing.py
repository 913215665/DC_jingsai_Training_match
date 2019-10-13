
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
# pd.set_option('display.max_rows', None)


def preprocessing(train, test):
    """
    数据预处理
    :param train:
    :param test:
    :return:
    """
    # 1.1 目标PM2.5值
    temp_target = pd.DataFrame()
    temp_target['pm2.5'] = train.pop('pm2.5')

    # 1.2 合并训练集 测试集
    data_all = pd.concat([train, test])
    data_all.reset_index(inplace=True)

    # 2 创建新特征
    # 2.1 年份 季度 月份 周数 每日 每日时间点(时)
    data_all['year'] = data_all['date'].apply(lambda x: x.year)
    data_all['quarter'] = data_all['date'].apply(lambda x: x.quarter)
    data_all['month'] = data_all['date'].apply(lambda x: x.month)
    data_all['week'] = data_all['date'].apply(lambda x: x.week)
    data_all['day'] = data_all['date'].apply(lambda x: x.day)

    # 2.2 是否下雪或降雨
    data_all['is_snow'] = np.zeros((data_all.shape[0], 1))
    data_all['is_rain'] = np.zeros((data_all.shape[0], 1))
    for i in range(data_all.shape[0]):
        if data_all['Is'][i] > 0:
            data_all['is_snow'][i] = 1

        if data_all['Ir'][i] > 0:
            data_all['is_rain'][i] = 1

    # 2.3 实际温度（t）与露点温度（Td），当t>Td时，表示空气未饱和，当t=Td时，已饱和，当t<Td时为过饱和
    # data_all['air_saturation_diff'] = data_all['TEMP'] - data_all['DEWP']
    # data_all['is_air_saturation'] = np.zeros((data_all.shape[0], 1))
    # for i in range(data_all.shape[0]):
    #     if data_all['air_saturation_diff'][i] <= 0:
    #         data_all['is_air_saturation'][i] = 1

    # 2.4 观测时间点对应的风速（m/s）
    # data_all['ws'] = np.zeros((data_all.shape[0], 1))
    # data_all['ws'][0] = data_all['Iws'][0]
    # for i in range(1, data_all.shape[0]):
    #     if data_all['Iws'][i-1] < data_all['Iws'][i]:
    #         data_all['ws'][i] = data_all['Iws'][i] - data_all['Iws'][i-1]
    #     else:
    #         data_all['ws'][i] = data_all['Iws'][i]

    # 4 离散化 二元化
    feats_dummy = [
                    'year', 'quarter', 'month', 'week', 'day', 'hour', 'is_snow', 'is_rain', 'Is', 'Ir'
                   ]
    temp_all = pd.get_dummies(data_all, columns=feats_dummy)

    # 5 特征选择
    temp_all.drop(['date', 'index'], axis=1, inplace=True)
    print(temp_all.columns.to_list())

    # 6 训练集  测试集划分
    temp_train = temp_all[temp_all.index < 35746]
    temp_test = temp_all[temp_all.index >= 35746]
    temp_train['pm2.5'] = temp_target['pm2.5']

    return temp_train, temp_test
