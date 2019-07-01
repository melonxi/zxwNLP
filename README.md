<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>
# zxwNLP

{{TOC}}

## 1 文本分类（情感分析）

### 1.1 LSTM_Attention模型在IMDB数据集下的文本分类的效果
  
#### 1.1.1 来源如下:  

* BiLSTM + Attention:  
[Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classiﬁcation](https://www.aclweb.org/anthology/P16-2034)  
* BiLSTM +  Feed Forward Networks with Attention:  
[Feed Forward Networks with Attention Can Solve Some Long-Term Memory Problems](https://arxiv.org/pdf/1512.08756.pdf)  
* BiLSTM + Hierarchical Attention:       
[Hierarchical Attention Networks for Document Classiﬁcation](https://www.aclweb.org/anthology/N16-1174) 
 
#### 1.1.2 性能比较如下：  

| Model Name | Accuracy | F1 |
| ------ | ------ | ------ |
| BiLSTM + Attention | 0.9075 | 0.9099 |
| BiLSTM + Feed Forward Networks with Attention | 0.9075 | 0.9094 |
| BiLSTM + Hierarchical Attention | 0.9065 | 0.9085 |
| BiLSTM | 0.8930 | 0.8940 |
| BiLSTM + 网络上某找不到原始论文的Attention| 0.8895 | 0.8895 |

#### 1.1.3 模型介绍如下：  
(个人理解)

* **BiLSTM + Attention**: （对应代码models.py中的BiLSTM_Attention） 

网络结构图：（来自论文）

<div align=center><img width="50%" height="50%" src="http://ww2.sinaimg.cn/large/006tNc79gy1g4kkqd87u0j314y0jo0yg.jpg"/></div >

Attention机制： 
 
${H}= {[h_1,h_2···h_t···h_T]}$ :LSTM所有时刻的隐状态输出组成的矩阵  
$$\begin{aligned} M &=\tanh (H) \\ \alpha &=\operatorname{softmax}\left(w^{T} M\right) \\ r &=H \alpha^{T} \end{aligned}$$

$Query = w$, 是一个 长为LSTM_outdim的向量。也即为Attention层学习参数，归纳了整个分类任务的关键信息。  
$Value = Key = h_t$, 通过内积求 $Query$ 和 $Value$ 的相似度，让网络了解应该重点关注句子的哪些时间步上的信息。

 
* **BiLSTM +  Feed Forward Networks with Attention**: （对应代码models.py中的BiLSTM_FFAttention）

网络结构图：（来自论文）

<div align=center><img width="50%" height="50%" src="http://ww4.sinaimg.cn/large/006tNc79gy1g4klcvtallj30q00tojv8.jpg"/></div >

Attention机制：

在实验中仅使用Word Attention 

$$\begin{aligned} u_{i t} &=\tanh \left(W_{w} h_{i t}+b_{w}\right) \\ \alpha_{i t} &=\frac{\exp \left(u_{i t}^{\top} u_{w}\right)}{\sum_{t} \exp \left(u_{i t}^{\top} u_{w}\right)} \\ s_{i} &=\sum_{t} \alpha_{i t} h_{i t} \end{aligned}$$

$Query = u_w$, 是一个 长为LSTM_outdim的向量。也即为Attention层学习参数，归纳了整个分类任务的关键信息。  
$Value = W_w{Key}+{b_w}\\Key=h_t$ 
通过学习得到参数，利用参数将 $Key$ 转化为 $value$ 
通过内积求 $Query$ 和 $Value$ 的相似度，让网络了解应该重点关注句子的哪些时间步上的信息。

* BiLSTM + Hierarchical Attention:（对应代码models.py中的BiLSTM_HAttention）

网络结构图：（来自论文）

<div align=center><img width="50%" height="50%" src="http://ww2.sinaimg.cn/large/006tNc79gy1g4klvqnfndj30ze0q2tbu.jpg"/></div >

$Value = Key = h_t$  
将学习到{uery}和利用{Query}求各个{Value}权重的过程都通过一个学习得到的MLP网络完成

#### 1.1.4 代码详情
## 目录结构描述 ##

├── Text Classification          //项目文件夹   
├──── data                //训练数据文件夹  
├────── labeledTrainData.tsv //训练数据  
├──── embedding        //预训练词向量文件夹  
├────── glove.840B.300d.txt //预训练词嵌入向量  
├──── config.py         //参数设置  
├──── dataPre.py                // 数据预处理  
├──── model.py              // 各网络模型  
└──── train.py        // 运行文件 

 注意：在glove.840B.300d.txt下的训练结果相较于glove.6B.300d.txt在FFAttention模型试验下有0.08的准确率提升，以上性能对比均在glove.840B.300d.txt下完成。
