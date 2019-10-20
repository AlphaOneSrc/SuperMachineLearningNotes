
### 模型
####逻辑回归
给定输入实例的特征向量$x$，则逻辑回归模型的输出可表示为$p(y=1|x)$。相应的有$p(y=0|x)=1-p(y=1|x)$。逻辑回归中需要训练的参数包括权重$W$和偏差项$b$。

$p(y=1|x)=\sigma(W^Tx+b)=(1+e^{-W^T-b})^{-1}$

下图中$x$轴表示$W^Tx+b$的值，$y$轴表示$p(y=1|x)$。(图片来自[wikipedia](https://en.wikipedia.org/wiki/Sigmoid_function))

<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/sigmoid.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;">sigmoid</div>
</center>

对于一个输入实例$(x^i,y^i)$，其损失函数为：

$LogLikelihood=\sum_{i=1}^{m}logP(y^i|x^i)=\sum_{i=1}^{m}log(\hat{y}^y(1-\hat{y})^{1-y})=-\sum_{i=1}^{m}L(\hat{y}^i,y^i)$

其中$\hat{y}^i$为预测值，$y^i$为真实标签。

对于全部训练数据而言，损失函数可以表示为：($m$是训练样本量)
$J(W,b)=\frac{1}{m}\sum_{i=1}^{m}L(\hat{y}^i,y^i)$

最小化损失函数实际上就等价于最小化似然函数：

$LogLikelihood=\sum_{i=1}^{m}logP(y^i|x^i)=\sum_{i=1}^{m}log(\hat{y}^y(1-\hat{y})^{1-y})=-\sum_{i=1}^{m}L(\hat{y}^i,y^i)$
</br>
####多分类(softmax回归)
softmax回归就是将逻辑回归(二分类)推广到多个类别的情形，即多分类模型。
<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/softmax_regression.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;">softmax</div>
</center>

</br>
如上图所示的一个3分类神经网络。在最后一层使用softmax激活函数输出每个类的概率。

softmax激活函数的数学描述如下所示：

1).$z^{[L]}=[z^{[L]}_0, z^{[L]}_1, z^{[L]}_2]$

2).$a^{[L]}=[\frac{e^{z^{[L]}_0}}{e^{z^{[L]}_0}+ e^{z^{[L]}_1}+e^{z^{[L]}_2}}, \frac{e^{z^{[L]}_1}}{e^{z^{[L]}_0}+e^{z^{[L]}_1}+ e^{z^{[L]}_2}}, \frac{e^{z^{[L]}_2}}{e^{z^{[L]}_0}+ e^{z^{[L]}_1}+e^{z^{[L]}_2}}]$

$=[p(class=0|x),p(class=1|x),p(class=2|x)]$
$=[y_0,y_1,y_2]$

损失函数为：
$LossFunction=\frac{1}{m}\sum_{i=1}^{m}L(\hat{y^i},y^i)$
$L(\hat{y},y)=-\sum_j^{3}y_j^i\log\hat{y_j^i}$

其中$m$是训练样本量，$j$表示第$j$个类。
</br>
####迁移学习
如果我们有较多的训练数据或者是我们的神经网络非常庞大的话，通常来说训练这样一个模型是非常耗时的（可能需要好几天或者好几周的时间）。值得庆幸的是，现在有非常多的预训练模型可供我们使用。通常，这些预训练模型都是在比较庞大的数据量上训练出来的。

迁移学习的主要思路就是我们可以下载这些预训练模型并且把它们调整用于我们自己的机器学习任务。如下图所示。

<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/transfer_learning.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;">迁移学习</div>
</center>
在数据量较多的情形下，我们可以重新训练整个神经网络。但如果我们可用的训练数据非常少的话，我们可以仅训练神经网络最后一层或者几层。

**在什么场景下我们可以使用迁移学习？**
假设：
预训练模型针对的是任务A，我们的模型针对的是任务B。

- 两个任务应该有着同样的输入形式
- 任务A有着大量的训练数据。但任务B的训练数据量非常少。
- 任务A学到的浅层特征对于训练任务B非常有帮助。

#### 多任务学习
如下所示，在一个分类任务中，通常每个输入实例有且仅有一个标签：
$y^{(i)} =\begin{pmatrix}0\\1\\0\\0\\0\end{pmatrix}$

但是在多任务学习中，一个输入实例可以有多个标签：
$y^{(i)} =\begin{pmatrix}0\\1\\1\\1\\0\end{pmatrix}$

在多任务学习中，损失函数可以写成：

$LossFunction=\frac{1}{m}\sum_{i=1}^{m}\sum_{j=1}^{5} L(\hat{y_i^j}, y_j^i)$

$L(\hat{y_j^i},y_j^i)=-y_j^i\log \hat{y_j}-(1-y_j^i)\log (1-y_j^i)$

其中$m$是训练样本量，$j$表示第$j$个类。

**多任务学习的一些Tips**
- 多任务学习可能会共享浅层特征
- 也许可以尝试训练一个足够大的神经网络，这个网络在所有任务上都表现较好
- 在训练集中，每个任务的输入实例是类似的

#### 卷积神经网络（CNN）
**滤波器/卷积核**
如下例所示，我们来看一个$3*3$的滤波器如何作用在一个$2D$的输入图像上。输入图像大小为$6*6$，那么卷积输出大小为$4*4$。

滤波器中的参数（如$w_1,w_2,...$）需要通过训练得到。
<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/cnn_on_2d.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;">CNN on 2D data</div>
</center>

另外，我们也可以同时使用多个滤波器叠加：
<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/cnn_on_2d_2_filters.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;">CNN on 3D data</div>
</center>

如果输入图像是三维的，我们也可以是使用$3D$卷积，在如下一个$3D$卷积中，我们有27个可学习的参数。
<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/cnn_on_3d_2_filters.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;">CNN on 3D data</div>
</center>

通常滤波器宽度都是奇数尺寸（如$1*1$，$3*3$，$5*5$，...）

滤波器的思想在于它对一部分输入起作用，可能另一部分输入也很起作用。 此外，卷积层输出值的每个输出值仅取决于少量输入。

**步长（stride）**
步长描述的是滤波器在输入图像上移动一次的大小，步长大小也决定了输出尺寸大小。
<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/cnn_stride.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;">padding</div>
</center>

值得注意的是，在做卷积运算时有些输入像素会被忽略掉，这个问题可以通过对输入添加填充来解决。

**填充（padding）**
根据是否添加padding卷积的方式可以分为两种：一种是不使用padding的valid卷积，一种是使用padding的same卷积。所谓填充，就是可以使用填充通过填充零来扩展原始输入，以便输出大小与输入大小相同。

如下例所示，输入大小为$6*6$，滤波器大小为$3*3$，如果设置步长为1、填充为1的话，我们可以得到与输入同样尺寸的输出。
<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/cnn_padding.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;">padding</div>
</center>

如果滤波器大小为$f*f$，输入大小为$n*n$，步长为$s$，则最终的输出大小为：
$(\lfloor \frac{n+2p-f}{s} \rfloor+1) \times (\lfloor \frac{n+2p-f}{s} \rfloor+1)$

**卷积层**
实际上，我们也可以像Relu激活函数一样对一个卷积层使用激活函数。
<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/cnn2_1_convolutional_layer.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;"></div>
</center>

对于卷积的参数量计算，以一个滤波器为例，有27个滤波器参数和1个偏置参数，所以总共有28个训练参数。

**1*1卷积**
使用$1*1$卷积的好处就是可以显著地减少计算量。不使用$1*1$卷积时的计算量：
<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/1_1_conv_1.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;"></div>
</center>

使用$1*1$时的计算量：
<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/1_1_conv_2.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;"></div>
</center>
    
**池化层（最大和平均池化）**
池化层可以看作是一种特殊的滤波器。

最大池化返回滤波器当前覆盖的区域的最大数值。类似的，平均池化返回该区域中所有像素值的平均值。

<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/cnn2_pooling_layer.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;">最大池化与平均池化</div>
</center>


在上图中，$f$为滤波器尺寸，$s$为步长大小。

**注意：池化操作不学习任何参数。**

**LeNet-5**
<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/cnn2_lenet_5.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;">LeNet-5</div>
</center>

LeNet-5大约有6万多的参数量。

**AlexNet**
<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/cnn2_alexnet.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;">AlexNet</div>
</center>

AlexNet参数量达到6千万。

**VGG-16**
<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/cnn2_vgg_16.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;">VGG-16</div>
</center>

VGG-16的参数规模达到1.38亿，模型中大量使用$3*3$卷积和$2*2$池化。

**ResNet**
<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/resnet.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;">ResNet</div>
</center>

$a^{[l+2]}=g(z^{[l+2]} + a^{[l]})$


**Inception**
<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/inception_network.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;">ResNet</div>
</center>


**目标检测**
- 分类和定位
<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/classification_with_localisation.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;">分类定位</div>
</center>


- 损失函数
<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/classification_with_localisation_1.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;">分类与定位损失函数</div>
</center>

- 地标检测
<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/classification_with_localisation_2.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;">地标检测</div>
</center>

- 滑动窗口检测算法
<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/sliding_windows.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;">分类器</div>
</center>

先用训练集训练出一个分类器，然后逐步将其用于目标图像。
<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/sliding_windows_1.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;">分类器</div>
</center>

为了计算损失（按顺序计算），我们可以使用滑动窗口的卷积来实现（即将最后完全连接的层转换为卷积层）。
<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/sliding_windows_2.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;">卷积分类器</div>
</center>

使用卷积实现，不需要按顺序计算结果。现在我们可以计算一次结果。
<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/sliding_windows_3.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;">执行卷积</div>
</center>


- 区域候选算法
实际上，很多时候我们只对一张图像中少数窗口感兴趣的对象。在区域候选（R-CNN）方法中，我们仅在建议的区域上执行分类器。

- R-CNN
  - 使用select search算法产生候选区域
  - 一次性地对这些候选区域进行分类
  - 预测标签值和边界框

- Fast R-CNN
  - 使用聚类方法来产生候选区域
  - 使用滑动卷积来对候选框进行分类
  - 预测标签值和边界框

  Faster R-CNN算法则是提出RPN（区域候选网络）来生成候选框。

  - YOLO算法
  - 预测边界框（YOLO算法的基础）
每张图片划分为很多个格子（cells）。
<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/yolo_1.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;">YOLO训练标签</div>
</center>

对于每个格子：
- $p_c$表示格子中是否存在目标物体
- $b_x$和$b_y$为[0,1]之间的点
- $b_h$和$b_w$为相对高度和宽度值
- $c_1$，$c_2$和$c_3$表示目标物体属于哪一类。

<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/yolo_2.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;">标签的组成部分</div>
</center>


- 交并比（IOU）
<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/yolo_3.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;">IOU</div>
</center>

按照惯例，我们使用0.5来定义阈值来判断预测的边界框是否正确。 例如，如果交并比大于0.5，则预测正确。

IOU也可以用来衡量两个边界框的相似程度。

- 非极大值抑制（NMS）
<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/yolo_4.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;">NMS</div>
</center>

算法可能会检测到多个物体，在上图中，两个边界框都检测到了猫、三个边界框都检测到了狗。NMS保证每个物体只检测到一次。

算法基本流程：
1）删除$p_c<0.6$的边界框
2）对于剩余的边界框：
$a$. 选择$p_c$值最大的边界框作为预测输出
$b$. 对剩余候选框，丢弃$IOU>0.5$的候选框，然后重复$a$。

- Anchor Boxes（锚定框）
前述算法只能检测一个单元格中的一个物体。但在某些情况下，单元格中有多个对象。为了解决这个问题，我们可以定义不同形状的边界框。

<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/anchor_box_1.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;">Anchor Box</div>
</center>

因此，训练图像中每个目标对象分配了：
- 包含对象中点的网格单元格
- 具有最大IOU的anchor box。

<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/anchor_box_2.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;">Anchor Box</div>
</center>

执行预测：
- 对于每个网格，我们可以获取两个预测的边界框
- 去除较低概率值的预测
- 对于每个类（$c_1,c_2,c_3$）使用NMS来生成最终预测结果。


**人脸验证**
- One-Shot学习（学习一个相似函数）
一次性学习：从一个例子中学习再次识别这个人。
函数$d(img1,img2)$表示img1和img2之间的差异程度。
<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/face_verification.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;">One-Shot Learning</div>
</center>

- Siamese网络 (学习相似/差异度)
<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/face_verification_1.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;">Siamese Network</div>
</center>

如果编码函数$f(x)$能够很好的表征一张图像，我们可以定义一个距离函数，如上图底端公式所示。

学习策略：
可训练参数：编码网络$f(x)$的参数
按照如下策略进行训练：
- 如果$x^{(i)}$和$x^{(j)}$是同一个人的话，令$||f(x^{(i)}) - f(x^{(j)})||^2$更小
- 如果$x^{(i)}$和$x^{(j)}$不算同一个人，令$||f(x^{(i)}) - f(x^{(j)})||^2$更大

- Triplet损失（同时看三张图片）
<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/face_verification_2.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;">Triplet Loss</div>
</center>

三张图片分别是：
- 锚定图片
- 正例图片：与锚定图片属于同一个人的图片
- 反例图片：与锚定图片不属于同一个人的图片

但直接学习上述损失函数可能会存在一个问题，即损失函数可能会导致$f(A)=f(P)=f(N)$。为了解决这个问题，我们可以给损失函数加一个小于零的数：
<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/face_verification_3.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;">Triplet Loss</div>
</center>

加总后的损失函数：
<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/face_verification_4.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;">Triplet Loss</div>
</center>

选择$A,P,N$三元组：
在训练时，如果对$A,P,N$随机取值，很容易满足$d(A,P) + \alpha \leq d(A,N)$，这时候训练算法很难奏效。我们可以选择相对较难训练的值：
<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/face_verification_5.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;">Triplet Loss</div>
</center>

- 人脸识别/验证和二分类
<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/face_verification_6.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;">二分类</div>
</center>

我们可以学习一个sigmoid二分类函数：
<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/face_verification_7.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;">二分类目标函数</div>
</center>

我们也可以使用其他变体，例如卡方相似度：
<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/face_verification_8.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;">卡方相似度函数</div>
</center>
   

**神经风格迁移**
<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/style_transfer_1.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;">风格迁移</div>
</center>

内容图片来自电影Bolt。
风格形象是百骏图的一部分，是中国最着名的古代绘画之一。
生成的图像由[https://deepart.io](https://deepart.io)支持。

损失函数包括两部分：$J_content$和$J_style$。
为了生成图像$G$:
1.随机初始化图像$G$
2.对$J(G)$执行梯度下降优化。

内容损失函数$J_{content}$：
内容损失函数用来确保原始图像的内容不被丢失。
1）使用隐藏层（不要太深也不要太浅）$l$来计算内容损失（我们可以从预训练的卷积网络中使用层$l$）
<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/style_transfer_2.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;">选择隐藏层</div>
</center>

2）
<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/style_transfer_3.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;">激活层</div>
</center>

3）
<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/style_transfer_4.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;">内容损失</div>
</center>

风格损失$J_{style}$：
1）使用$l$层激活函数来衡量风格。
<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/style_transfer_5.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;">选择隐藏层</div>
</center>

2）将图像的风格定义为跨通道激活之间的相关性
<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/style_transfer_6.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;">隐藏层的通道</div>
</center>

矩阵中的元素$G$反映了不同通道之间的激活的相关性。
对于风格图像：
<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/style_transfer_7.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;">风格图像矩阵</div>
</center>

对于生成图像：
<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/style_transfer_8.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;">生成图像矩阵G</div>
</center>

风格函数：
<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/style_transfer_9.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;">风格函数</div>
</center>

你也可以考虑关联不同层之间的风格损失：
<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/style_transfer_10.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;">关联不同层之间的风格损失</div>
</center>

**1D和3D卷积**
<center>
    <img 
    src="https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/1d_and_3d_generalisations.png">
    <br>
    <div style="color: #999;
    font-size:11px;
    padding: 2px;">1D和3D卷积</div>
</center>
