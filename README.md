# zxwNLP
项目对比了LSTM+各类attention机制在IMDB数据集下的文本分类的效果  
来源如下：  
[Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classiﬁcation](https://www.aclweb.org/anthology/P16-2034)
[Feed Forward Networks with Attention Can Solve Some Long-Term Memory Problems](https://arxiv.org/pdf/1512.08756.pdf)  
[Hierarchical Attention Networks for Document Classiﬁcation](https://www.aclweb.org/anthology/N16-1174)

| Model Name | Accuracy | F1 |
| ------ | ------ | ------ |
| BiLSTM + Attention | 0.9075 | 0.9099 |
| BiLSTM + Feed Forward Networks with Attention | 0.9075 | 0.9094 |
| BiLSTM + Hierarchical Attention | 0.9065 | 0.9085 |
| BiLSTM | 0.8930 | 0.8940 |
| BiLSTM + 网络上某找不到原始论文的Attention| 0.8895 | 0.8895 |

