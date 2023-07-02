# self-supervised-model
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在CIFAR-10的训练集上使用ResNet-18网络架构，进行自监督的预训练，自监督学习方法选用CPCModel，
并在验证集上进行评价指标的测试，最后在测试集上进行Linear Classification Protocol。
<br>
## 一、数据处理
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;CIFAR-10数据集包含60000张图片（32×32的像素），共10个类别，每一个子类为600张图片。
CIFAR-10划分为训练集样本45000，验证集样本5000，测试集样本10000。<br>
<br>
&nbsp;&nbsp;&nbsp;&nbsp;数据预处理：<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.进行填充，padding参数取4<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.随机水平翻转<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.标准化Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))，使数据分布为[-1,1]<br>

## 二、模型设定
（一）自监督模型<br>
&nbsp;&nbsp;&nbsp;&nbsp;本文使用ResNet-18 架构作为编码器，选择CPCModel作为自监督学习方法进行预训练，并添加了投影头和分类器。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.将原有的全连接层替换成了恒等映射，去除了原来的分类层，这是因为自监督学习的目标不需要具体的标签<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.定义了一个投影头来进一步提取编码器输出的特征。投影头由两个线性层组成，大小分别是(512, 256) 和 (256, 128)，从而将输入的特征从 512 维逐层映射到128维，中间使用ReLU作为激活函数<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.定义线性分类器，大小为(128，10)<br>

（二）Linear Classification Protocol<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;本文在线性分类器中，主要定义了一个线性层，将输入特征从512维映射到10维，在前向传播中，把输入特征展平成一维向量，再通过线性层进行线性变换。<br>

（三）损失函数和优化器<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.损失函数为交叉熵损失函数<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.采用SGD优化器，学习率为0.001，动量参数为0.9，权重衰减参数为0.0001<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.学习率调度器，使用减小学习率的回调函数，并用验证集的损失来调整学习率<br>
<br>

## 三、模型运行
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;batch_size设置为128，epoch设置为20，共7000多个iteration<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.model.py为在CIFAR-10上做resnet-18的自监督模型，并进行Linear Classification Protocol<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.baseline.py为在CIFAR-10做resnet-18分类<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;自监督模型在baseline基础上提升了3.7%的分类准确率<br>

