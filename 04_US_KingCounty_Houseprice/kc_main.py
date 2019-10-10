# coding:utf-8

"""
    任务
    我们希望学完《数据分析师（入门）》的学员，可以根据课上老师所讲授的知识和回归分析的方法，
    从给定的房屋基本信息以及房屋销售信息等，建立一个回归模型预测房屋的销售价格。

    数据
    *注 :报名参赛或加入队伍后，可获取数据下载权限。
    数据主要包括2014年5月至2015年5月美国King County的房屋销售价格以及房屋的基本信息。
    数据分为训练数据和测试数据，分别保存在kc_train.csv和kc_test.csv两个文件中。
    其中训练数据主要包括10000条记录，14个字段，主要字段说明如下：
    第一列“销售日期”：2014年5月到2015年5月房屋出售时的日期
    第二列“销售价格”：房屋交易价格，单位为美元，是目标预测值
    第三列“卧室数”：房屋中的卧室数目
    第四列“浴室数”：房屋中的浴室数目
    第五列“房屋面积”：房屋里的生活面积
    第六列“停车面积”：停车坪的面积
    第七列“楼层数”：房屋的楼层数
    第八列“房屋评分”：King County房屋评分系统对房屋的总体评分
    第九列“建筑面积”：除了地下室之外的房屋建筑面积
    第十列“地下室面积”：地下室的面积
    第十一列“建筑年份”：房屋建成的年份
    第十二列“修复年份”：房屋上次修复的年份
    第十三列"纬度"：房屋所在纬度
    第十四列“经度”：房屋所在经度

    测试数据主要包括3000条记录，13个字段，跟训练数据的不同是测试数据并不包括房屋销售价格，
    学员需要通过由训练数据所建立的模型以及所给的测试数据，得出测试数据相应的房屋销售价格预测值。

    （注：比赛所用到的数据取自于kaggle datasets，由@harlfoxem提供并分享。我们只选取了其中的子集，
    并对数据做了一些预处理使数据更加符合回归分析比赛的要求。）

    如遇数据下载打开乱码问题：
    不要用excel打开,用notepad++或者vs code。文件格式是通用的编码方式utf-8，如果要用excel,请转换为ansl格式或者gbk格式。

    评分标准
    评分算法
    regression
    算法通过计算平均预测误差来衡量回归模型的优劣。平均预测误差越小，说明回归模型越好。平均预测误差计算公式如下：
    mse = 1/10000m * Σ(pred_y - y)²
    其中，mse是平均预测误差，m是测试数据的记录数（即3000），是参赛者提交的房屋预测价格，y是对应房屋的真实销售价格。
    说明：为更好地显示排行榜成绩，DC算法工程师统一将参赛选手提交的得分除以10000后得到的数值
    （即m前面的系数10000），显示在最后的排行榜上。
"""

import time
from kc_data_import import read_data
from kc_data_preprocessing import preprocessing
from kc_data_prediction import predict


def main():
    # 读取数据
    columns_test = ['date', 'bedroom', 'bathroom', 'floor space', 'parking space', 'floor', 'grade',
                     'covered area', 'basement area', 'build year', 'repair year', 'longitude', 'latitude']
    columns_train = ['date', 'price', 'bedroom', 'bathroom', 'floor space', 'parking space', 'floor', 'grade',
                     'covered area', 'basement area', 'build year', 'repair year', 'longitude', 'latitude']
    test = read_data('kc_test.csv', columns_test)
    train = read_data('kc_train.csv', columns_train)

    # 数据预处理
    train_data, test_data = preprocessing(train, test)

    # 模型预测评估 并 输出预测数据
    pred_y = predict(train_data, test_data, is_shuffle=False)
    pred_y.to_csv('./kc_pred_0925.csv', index=False, header=['price'])


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()

    print("耗时：", end_time - start_time)