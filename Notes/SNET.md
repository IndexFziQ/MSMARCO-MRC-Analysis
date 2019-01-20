# My Notes -- From Answer Extraction to Answer Generation for Machine Reading Comprehension (S-Net)




## Introduction

本文出自MSRA，一作是北航的Chuanqi Tan, 代码没有公开，链接为台湾的一位同学用*CNTK*实现的代码。首次使用了*Seq2Seq*模型对*MS-MARCO*阅读理解数据集进行答案生成，比较符合*MARCO*数据集的本意，同时定义了先抽取在生成的框架，在*R-Net*的基础上有一定的提升。

<!--more-->

模型在*MS-MARCO V1*的排名是第三，效果与抽取式的模型结构并没有太大的提升，这也让人怀疑*Seq2Seq*的作用到哪里去了。文章有几处疑点，一是*Attention Pooling*时，公式有些问题；二是*Seq2Seq*的输入到底是什么，如果出现答案来自于多篇文章的情况，输入还是按照答案-文档对来做的吗？

## Motivation
这篇文章的动机就是抽取式的网络结构并不适合微软阅读理解数据集，有很多情况下，答案需要根据问题和文档组合或者生成。如图所示，除了第一种情况，其他都不能简单的抽出来。
    <div align=center>
       <img src="./images/SNET/motivation.png" height="50%" width="50%" />
      </div>

## Contribution
* 为多篇章阅读理解（或者说生成式）提出了一个先抽取再生成的框架；
* 在抽取的模块，采用多任务学习，利用*Page ranking*这个辅助任务帮助抽取答案片段；
* 第一个将序列到序列模型应用到阅读理解数据集。


## Model

下面对模型详细介绍：模型整体比较复杂，大体就是*R-Net*和*Seq2Seq*的组合。

### Overview

<div align=center>
       <img src="./images/SNET/S-Net-F.png" height="50%" width="50%" />
      </div>

由上图可以看出，整个模型是一个*pipeline*结构，并不是端到端的，*Seq2Seq*的效果本来就不是很好，因此第一个模块抽取的质量很关键，能把*Seq2Seq*用上本身就是一种突破。

模型分两个模块，分别是*Evidence Extraction Model*和*Answer Synthesis Model*。

### Evidence Extraction Model
<div align=center>
       <img src="./images/SNET/S-Net-E.png" height="50%" width="50%" />
      </div>

这个模型从本质上讲和*R-Net*有很大的相似性，只是没有高层的*self-attention*。

**Embedding Layer**

该层的输入就两个，词级别的表示和字符级别的表示，然后喂给双向GRU得到最终的表示 $U^Q_{t}$，$U^P_{t}$。

**Question-aware passage representation**

这部分就是*passage*对*query*做*attention*，得到权重化的表示，也就是所谓的*Question-aware*。这部分较复杂，过程简单来讲就是计算*attention*，过门控函数，再用GRU得到表示$V^P_t$。

之后还是一个计算*attention*的过程，$U^Q_{t}$做一个*self-attention*，得到最终的表示$r^Q$。然后用得到的问题表示$r^Q$与GRU得到表示$V^P_t$做*attention*，加权求和得到最终的*passage*表示$r^P$。

这里可以提一下多任务学习中的硬共享模式，这两个表示也就硬共享模式中的*shared representation*，在此基础上，在根据任务在最后一层有不同的输出。任务分别是*Evidence Prediction*和*Page Ranking*。

**Evidence Prediction**

主要思想就是指针网络，求*span*起始位置和终止位置的概率值。

首先用问题最后的表示$r^Q$初始化*GRU*的第0时刻的隐藏层状态$h^a_0$，然后就是标准的*pointer network*。输入是$r^Q$和$V^P$，*GRU*之后过*softmax*，选择概率最大的作为输出。

**Page Ranking**

这个任务本质上是个二分类问题，合理利用了数据集中给的标注，问题的答案是否用到当前*passage*的内容，用到标为1，没用到标为0。直接把$r^Q$和$r^P$喂给分类器就可以了，任务简单，但是给了抽取任务一定的监督信号，有一定的辅助作用。

**Training for Evidence Extraction Model**

抽取模块的损失函数分两部分，用超参去调相应的比重。
$${L}_{E} = \lambda{L}_{AP}+{(1-\lambda)}_{PR}$$

### Answer Synthesis Model
<div align=center>
       <img src="./images/SNET/S-Net-A.png" height="50%" width="50%" />
      </div>

抽取片段的模型部分得到*evidence*，之后并不是直接就作为*Seq2Seq*的输入，中间还有一定的处理过程。

**Initialization**

首先，用*BiGRU*得到问题和文档的表示，文档的表示是由抽取模块处理好的*passage*（把起始位置和终止位置作为0，1特征）作为输入。
$$h_t^P = BiGRU(h_{t-1}^P,[e_t^P,f_t^s,f_t^e])$$
$$h_t^Q = BiGRU(h_{t-1}^Q,e_t^Q)$$
其中，$f$用来表示*span*的起止位置，然后$d_0$初始化*Seq2Seq*，其中：
$$d_0 = tanh(W_d[h_1^P,h_1^Q] + b)$$
这里就是取*BiGRU*的第一个隐藏层状态作为MLP的输入得到*Seq2Seq*所需要的$d_0$。


**Answer Synthesis**

答案合成这个部分就是标准的*Attention-based Seq2Seq*，图上画的也比较清楚了，公式如下
$$d_t = GRU(w_{t-1},c_{t-1},d_{t-1})$$
其中*Attention*是的计算就是最普通的*Global Attention*。最终的输入在过*Maxout*激活函数和*softMax*，即得到概率分布。

**Training for Answer Synthesis Model**

答案合成模块的损失函数就是负对数似然。

## Experiments

<div align=center>
       <img src="./images/SNET/experiment.png" height="50%" width="50%" />
      </div>

从实验结果上可以看出，*S-Net*的答案抽取模块并没有*R-Net*好，虽然加了一个*Passage Ranking*的辅助任务，但是还是没有对*Passage representation*最后再做一次*self-attention*效果好。不过，最后加上Seq2Seq效果好很多，接近人类的水平，个人观点是在抽取span之后处理的好，当然不可否认可以把Seq2Seq训练好也不容易了。


## Conclusion

最后，对S-Net做一个简单的总结：
1. 为多篇章（生成式）阅读理解提供了一个先抽取再生成的框架;
2. 使用多任务学习辅助训练主任务;
3. 第一次将*Seq2Seq*应用到阅读理解的任务中。

## References

1. S-Net: From Answer Extraction to Answer Generation for Machine Reading Comprehension. JChuanqi Tan, Furu Wei, Nan Yang, Bowen Du, Weifeng Lv and Ming Zhou. AAAI 2018. 
2. MAMARCO leaderboard: http://www.msmarco.org/leaders.aspx
3. MAMARCO Analysis: https://github.com/IndexFziQ/MSMARCO-MRC-Analysis