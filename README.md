# 2017腾讯鼠标轨迹数据人机识别
队伍名称：DutyFree Fish，初赛13名，复赛68名

# 初赛数据下载：
链接：https://pan.baidu.com/s/1zIA31lUPhV3K9yCONBIGRw  提取码：t8is 
训练数据：dsjtzs_txfz_training.txt
线上测试数据A榜：dsjtzs_txfz_test1.txt
线上测试数据B榜：dsjtzs_txfz_testB.txt

# 初赛
1）比赛初期，将轨迹数据图像化，使用CNN模型，线上F1值0.7726，效果不好

2）后来使用LGB模型，经过特征提取与筛选，最终线下F1值0.9617，线上B榜F1值0.9070

3）尝试过CNN模型卷积特征输出，与LGB模型融合，但效果不好

# 复赛
线上赛，使用腾讯大数据计算平台，200万条真实数据
1）pyspark提取特征，见复赛文件夹下extra_feature.py文件

2）线上XGB模型、LR模型融合


比赛总结：https://blog.csdn.net/Aseri_ldn/article/details/77449101
