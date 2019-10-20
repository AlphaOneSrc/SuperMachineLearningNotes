### 正则化

在机器学习中，正则化是一种避免过拟合的常用方法。通过添加正则化项到损失函数中实现。

#### - L2 正则化

$$ \min J(W,b)=\frac{1}{m}\sum_{i=1}^mL(\hat{y}^i,y^i)+\frac{\lambda}{2m}||W||_2^2 $$

在新的损失函数中, $\frac{\lambda}{2m}||W||_2^2$ 是正则化项 ， $\lambda$ 是正则化参数 (一项超参数). L2 正则化 也被称作权重衰减。

对于逻辑回归模型, $W$ 是一个向量 (例如 $W$ 的维度同特征向量的一致), 因此，正则化项也可以写为如下:

$||W||_{2}^2=\sum_{j=1}^{dimension}W_{j}^2$.

对于多层神经网络模型 (如 含 $L$ 层), 层间将会形成多个参数矩阵. 每个矩阵的形状​为 $(n^{[l]}, n^{[l-1]})$.

在上式中, $l$ 表示神经网络的第 $ l $ 层 ， $n^{[l]}$ 表示第 $l$ 层中神经元的个数.  因此， L2 正则项表示为:

$\frac{\lambda}{2m}\sum_{l=1}^L||W^l||_2^2$

$||W^l||_{2}^2=\sum_{i=1}^{n^{[l-1]}}\sum_{j=1}^{n^{[l]}}(W_{ij}^l)^2$ (又称 F范数).

#### - L1 正则

$\min J(W,b)=\frac{1}{m}\sum_{i=1}^mL(\hat{y}^i,y^i)+\frac{\lambda}{2m}||W^l||$

$||W^l||=\sum_{i=1}^{n^{[l-1]}}\sum_{j=1}^{n^{[l]}}W_{ij}^l$.

使用 L1 正则, 参数 $W$ 将会是稀疏的

#### - Dropout (inverted dropout 注吴恩达讲解的方法)

直觉上理解 dropout, 它也看做是一个正则化项，目标是为了使得监督模型更加健壮。通常在训练阶段，dropout的实现是通过忽略部分神经元的输出值。因此，在做预测时，模型将要不仅仅依赖任何一个特征。 

在dropout正则化中，超参数“保留概率”（keep probability）描述激活单元被选用的概率。因此，假如一个隐层有 $n$ 个单元，保留概率为$p$, 则大约有 $p \times n$ 个单元被激活，剩余的$(1-p)\times n$ 单元会被忽略。

例如:**
![dropout example](https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/dropout.png)

如上所示，第2层的2个单元被忽略掉后，第三层的线性组合值将会变 $z^{[3]=}W^{[3]}a^{[2]}+b^{[3]}$，并且此值会变小。为了不降低$z$值，我们应该调整 $a^{[2]}$ 的值，方法为除以一个保留概率值，也就是 $a^{[2]} := \frac{a^{[2]}}{p}$

记住：做预测时，没有必要使用dropout正则。

#### - 早停法

同样使用早停法是为了防止模型出现过拟合。
![early stopping](https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/early_stopping.png)

