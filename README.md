# SuperMachineLearningNotes
Super-Machine-Learning-Revision-Notes

#### 词嵌入（Word embedding）

##### One-hot

![One-Hot](https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/one_hot.png)

##### 嵌入矩阵（Embedding Matrix） $E$

![Embedding Matrix](https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/embedding_matrix.png)

$ UNK $表示未知词，所有识别到不在词库里面的单词都会强制转换为$ UNK $。 矩阵由$ E $表示。 如果我们想获取一个单词的词嵌入向量，我们可以如下图所示利用单词的one-hot向量进行转化：

![Get Word Embedding](https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/get_embedding.png)

通常，可以将其公式化为：
$$
E \cdot O_j = e_j (embedding \quad for \quad j )
$$

##### 学习词嵌入（Learning Word Embedding）

![Learning Word Embedding](https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/learning_word_embedding.png)

在模型中，嵌入矩阵（即$E$）与其他参数（即$w$和$b$）一样是可学习的。 所有可学习的参数都以蓝色突出显示。

该模型的总体思想是在给定上下文的情况下预测目标单词。 在上图中，上下文是最后4个单词（即a，glass，of，橙色），目标单词是“ to”。

另外，有多种方法可以定义目标词的上下文，例如：

- 最后$n$个字
- $n$个单词左右目标单词
- 附近的一个字（skip-gram的思想）
- …

##### Word2Vec & Skip-gram

假设有一个句子“I want a glass of orange juice to go along with my cereal.”

在此词嵌入学习模型中，上下文（context）是从句子中随机选择的词，目标（target）是用上下文单词的窗口随机拾取的单词。

例如，我们现在将“orange”作为上下文单词，我们能得到下面的训练样本：

![Context and Target](https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/skip_gram_context_target.png)

模型是这样的：

![Model](https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/skip_gram_model.png)

Softmax函数可以被定义为如下形式：

![Softmax](https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/word_embedding_softmax.png)

$\theta$是与输出有关的一个参数，$e_c$是当前上下文单词的embedding形式。

然而使用softmax函数的问题分母的计算成本太大，原因是我们的词汇量可能很大。 为了减少计算量，负采样是不错的解决方案。

##### 负采样（Negative Sampling）

假设有一个句子“I want a glass of orange juice to go along with my cereal.”

给定一对单词（即上下文单词和另一个单词）和标签（即第二个单词是否为目标单词）。 如下图所示，（orange juice 1）是一个正样本，因为单词juice是orange的真正目标单词。 由于所有其他单词都是从词典中随机选择的，因此这些单词被视为错误的目标单词。 因此，这些对是负样本（偶然将真实的目标单词选作否定示例是可以接受的）。

![Negative Sampling](https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/negative_sampling.png)

至于每个上下文单词对应的错误单词的数量$K$，在样本量较少的情况下，$K$选择在5-20之间，如果样本量较大，$K$选择在2-5之间。

模型如下：

![Negative Sampling Model](https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/negative_sampling_model.png)

我们只能训练softmax函数的$K+1$个逻辑斯蒂回归模型，所以计算量会低很多。

那么如何筛选负样本呢？模型如下：

![Sampling Distribution](https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/sample_word_distribution.png)

$f(w_i)$是词频。

如果使用第一个样本分布，则可能总是选择诸如 “the”、“a”、“of”等之类的词。但是，如果使用第三个分布，则所选的词将是非代表性的。 因此，第二分布可以被认为是用于采样的更好的分布。 这种分布在第一个和第三个之间。

##### GloVe 向量

符号: $X_{ij}= 词汇_i出现在上下文j的数量$。

模型如下：

![Objective Function](https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/glove.png)

$X_{ij}$衡量两个词的相关性以及两个词同时出现的频率（共现性）。$f(X_{ij})$是权重模块，它给高频词汇对（两两匹配衡量$X_{ij}$）带来了不太高的权重，也给了不太常见的词汇对带来了不太小的权重。

如果我们仔细看$\theta$和$e$，会发现其实他们起到的作用一样，所以最终单词的词嵌入形式为：

![Final Word Embedding](https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/glove_final_embedding.png)

##### 深度上下文化的词表示形式（ELMo）

*预训练双向语言模型*

前向语言模型：给地方给你一个长度为$N$D的序列$(t_1,t_2,...,t_N)$，前向语言模型通过对$t_k$的概率来建模计算序列的概率，即：

![img](https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/forward_language_model.jpg)

后向语言模型类似：

![img](https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/backward_langauge_model.jpg)

双向语言模型融合了上述两者，同时最大化前向和后向概率。

![img](https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/bi_language_model.jpg)

LSTM被用于前向和后向语言模型建模中。

![bidirectional language model](https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/biLM.png)

就输入嵌入而言，我们可以只初始化这些嵌入或使用预先训练的嵌入。 对于ELMo，使用字符嵌入和卷积层会更加复杂，如下所示。

![Input Embeddings](https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/input-embedding.png)

训练语言模型后，我们可以将得到单词在这个句子中的ELMo嵌入。

![ELMo](https://createmomo.github.io/2018/01/23/Super-Machine-Learning-Revision-Notes/elmo.png)

在ELMo中，$s$是经过softmox标准化的权重，$\gamma$是一个能根据任务模型缩放整个ELMo的缩放向量。这些参数可以在特定任务模型训练期间进行训练。

参考文献：
[1] https://www.slideshare.net/shuntaroy/a-review-of-deep-contextualized-word-representations-peters-2018
[2] http://jalammar.github.io/illustrated-bert/
[3] https://www.mihaileric.com/posts/deep-contextualized-word-representations-elmo/






