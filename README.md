# Mobile-Classify
***
##数据集介绍

数据集包含2000*21维数据，最后一维数据为目标变量价格情况，0表示低成本，1表示中成本，2表示高成本，3表示非常高成本。
属性变量包含前20维，分别包含电池容量、是否有蓝牙、微处理器执行指令的速度、是否支持双卡、前置摄像头像素、后置摄像头像素等等属性变量。

##定义网络模型

采用两层全连接网络，最后一层softmax输出层

##获取数据

拆分x_train,x_test,y_train,y_test数据
转换数据形态

##定义训练模型

设置学习率参数0.001，迭代次数1500

##定义测试

找到测试结果中概率最大的下标，