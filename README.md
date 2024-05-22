# MNIST_Training_MyDataset

通过继承Dataset类来构建自己的数据集My_Dataset来实现自定义数据集
通过重写
+ `__init__()`来通过传入地址，是否为训练集，已经transform来获取对应目录的数据集
+ `__getitem__()`来获取对应下标的`(tenor图片,标签)`二元组
+ `__len__()`，来获取数据集总长度
