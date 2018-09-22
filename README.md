# deep_learning
在大部分的情况下，逻辑回归算法的效果除了依赖于训练数据，也依赖于从数据中提取的特征。假设我们从邮件中抽取的特征只有邮件发送的时间，那么即使有再多的训练数据，逻辑回归算法也无法很好地利用。

深度学习解决的核心问题之一就是自动的将简单的特征组合成更加复杂的特征，并使用这些组合特征解决问题。深度学习除了可以学习特征和任务之间的关联以外，还能自动从简单特征中提取更加复杂的特征。  
[张量（tensor）：](https://github.com/xinchen201795/deep_learning/blob/master/tensorflow1.ipynb)  
一个张量主要保存了三个属性：姓名（name）,维度（shape）和类型（type）
* 命名：node:src_output node为计算节点的名称，src_output表示来自节点的第几个输出（编号从0开始）。 
* shape =(2.)说明是一个一维数组，这个数组的长度为2.
* 类型，每个张量的类型唯一，运算时不匹配会报错
* 会话，session用于执行定义好的运算
神经元结构
## 全连接神经网络
相邻两层之间任意两个节点之间都有连接  
## [tensorflow的前向传播](https://github.com/xinchen201795/deep_learning/blob/master/tensorflow前向传播.ipynb)
神经网络的前向传播需要三部分信息：  
1. 神经网络的输入 从实体中提取的特征向量
2. 神经网络的连接结构，在tensorflow中通过矩阵乘法实现神经网络的前向传播过程：  
a = tf.nn.relu(tf.matmul(x,w1)+b1)  
y = tf.nn.relu(tf.matmul(a,w2)+b2)  
没有定义w1,w2,b1,b2  
TensorFlow通过变量（tf,Variable）来保存和更新神经网络中的参数，比如定义w1：
weights = tf.Variable(tf.random_normal([2,3],stddev = 2))  
tf.random_normal([2,3],stddev = 2)会产生一个2* 3的矩阵，矩阵中的元素均值为0，标准差为2
## tensorflow的反向传播
![tensorflow的反向传播](https://github.com/xinchen201795/deep_learning/blob/master/反向传播.png)
在每次迭代的开始，首先需要选取一小部分训练数据，这一小部分数据叫做一个batch。基于预测值和真实值之间的差距，反向传播算法会相应更新神经网络参数的取值，使得在这个batch上神经网络模型的预测结果与真实答案更接近。
* placeholder定义一个需要指定类型的位置，不需要增加节点计算  
[placeholder前向传播](https://github.com/xinchen201795/deep_learning/blob/master/placeholder前向传播.ipynb)  [损失函数](https://github.com/xinchen201795/deep_learning/blob/master/损失函数.ipynb)  
比较常用的优化方法有:tf.train.GradientDescentOptimizer,class tf.train.AdamOptimizer和tf.train.MomentumOptimizer.
