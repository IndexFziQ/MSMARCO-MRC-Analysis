# Analysis on MSMARCO MRC Submissions

A list of analysis on the MS-MARCO leaderboard regarding the machine reading comprehension task.

Contributed by Yuqiang Xie and Xing Luxi, *National Engineering Laboratory for Information Security Technologies, Institute of Information Engineering, Chinese Academy of Sciences, Beijing, China*.

## Introduction

[MS MARCO](https://arxiv.org/pdf/1611.09268.pdf) (Microsoft Machine Reading Comprehension) is a large scale dataset focused on machine reading comprehension, question answering, and passage ranking. The current dataset has 1,010,916 unique real queries that were generated by sampling and anonymizing Bing usage logs. For more details, see the project: [MSMARCOV2](https://github.com/dfcf93/MSMARCOV2).

**MSMARCO** is a benchmark dataset for multi-passage MRC or so-called "generative MRC". Besides, different from other MRC datasets, MSMARCO has the following advantages (referred from MSMARCOV2):

1. Real questions: All questions have been sample from real anonymized bing queries.
2. Real Documents: Most Url's that we have source the passages from contain the full web documents (abstract). These can be used as extra contextual information to improve systems or be used to compete in our expert task.
3. Human Generated Answers: All questions have an answer written by a human (not from a passage span). If there was no answer in the passages the judge read they have written 'No Answer Present.'
4. Human Generated Well-Formed: Some questions contain extra human evaluation to create well formed answers that could be used by intelligent agents like Cortana, Siri, Google Assistant, and Alexa.
5. Dataset Size: At over 1 million queries the dataset is large enough to train the most complex systems and also sample the data for specific applications.

In this repository, submissions to MSMARCO will be almost listed.  This work will be updated persistently.

Tips: The official evaluation metrics include [ROUGE-L](http://aclweb.org/anthology/W04-1013) and [BLEU-1](http://www.anthology.aclweb.org/P/P02/P02-1040.pdf). 

## V1 Leaderboard - Model List

Task Definition: Given a question and a set of passages (top 10) retrieved
by search engines, the task is to find the best concise answer to the question. 

|Rank|Model| Org. | Rouge-L | Bleu-1 | 
|:---:|:----|:-------|:-----:|:-----:|
|1|[MARS](#MARS)| YUANFUDAO research NLP | 49.72| 48.02 | 
|2|Human Performance| -  |47.00| 46.00 |-|
|3|[V-Net](#V-Net)| Baidu NLP | 46.15 | 44.46 |
|4|[S-Net](#S-Net)| Microsoft AI and Research |45.23| 43.78|
|5|[R-Net](#R-Net)| Microsoft AI and Research |42.89| 42.22|
|6|[ReasoNet](#ReasoNet)| Microsoft AI and Research |38.81| 39.86|
|7|[Prediction](#Prediction)| Singapore Management University |37.67| 33.93|
|8|[FastQA_Ext](#FastQA_Ext)| DFKI German Research Center for AI |33.67| 33.93|


## V2 Leaderboard - Model List

For this version, we focus on *Q&A + Natural Langauge Generation*, where human performance is 63.21 on Rouge-L.

Task Definition: Given a query and 10 passages provide the best answer avaible in natural languauge that could be used by a smart device/digital assistant.

|Rank| Model      | Org.  | Rouge-L | Bleu-1 |
| :--------: | :---- | :----------- | :--------: | :------: |
|1| Human Performance    | -   |      63.21        |   53.03  |
|2| V-Net  | Baidu NLP |  48.37		 | 46.75 |
|3| Masque | NTT Media Intelligence Laboratories| 46.81 |47.64 |
|4| SNET + CES2S   | SYSU     |    45.04          |  40.62  |
|5| Reader-Writer   |  Microsoft Business Applications Group AI Research     |    43.89          |    42.59        |
|6| ConZNet |   Samsung Research    |    42.14          | 38.62 |

## Description of Models

### <span id = "MARS">MARS</span>
**Multi-Attention ReaderS Network.** *Jingming Liu.* [ [video](https://v.qq.com/x/page/k06284mr0hk.html) ]
* **Motivation**
    * Transfer learning tasks like CoVe and ELMo store more generalized information in the lower layer (encoder).
    * The answers from each site are quite different and have their own features.
    * Multi-task learning may bring an improvement for the MRC task.
    * The score for multi-passage needs more factors, not just the simple counting.
    * Through training, the train set is large and the batch is small, which leads to the distribution between real samples and each batch inconsistent. The output of model is related to the order of the batch.
* **Contribution**
    * Enhance Glove with CoVe, ELMo, POS, NER and word-match features, word dropout is 5%. *[ nearly 2.0 improvement ]*
    * Embed each site to a representation and combine it with the passage representation in the prediction layer. *[ nearly 0.3-0.5 improvement ]*
    * Apply multi-task learning into MRC. 
        * Main task: 
            * Golden span (the highest rouge(>0.8) span in every passage). 
        * Auxiliary task: *[ weight: 0.1~0.2 ]*
            * If a word is in the answer;
            * If a passage contains a golden span;
            * If a sentence contains a golden span.
    * The score of answer contains span score and vote score.
    * Introduce EMA (Exponential Moving Average) into the training for weakening the effect of batch order. *[ nearly 0.3 improvement ]*
* **Overview**

![MARC](mars.png)

### <span id = "V-Net">V-Net</span>
**Multi-Passage Machine Reading Comprehension with Cross-Passage Answer Verification.** *Yizhong Wang, Kai Liu, Jing Liu, Wei He, Yajuan Lyu, Hua Wu, Sujian Li and Haifeng Wang.*  ACL 2018. [ [pdf](http://aclweb.org/anthology/P18-1178) ]
* **Motivation**
* **Contribution**


### <span id = "S-Net">S-Net</span>
**S-Net: From Answer Extraction to Answer Generation for Machine Reading Comprehension.** *JChuanqi Tan, Furu Wei, Nan Yang, Bowen Du, Weifeng Lv and Ming Zhou.* ArXiv 2017. [ [pdf](https://arxiv.org/pdf/1706.04815.pdf) ]
* **Motivation**
* **Contribution**

### <span id = "R-Net">R-Net</span>
**Gated Self-Matching Networks for Reading Comprehension and Question Answering.** *Wenhui Wang and Nan Yang and Furu Wei and Baobao Chang and Ming Zhou.* ACL 2017. [ [pdf](http://aclweb.org/anthology/P17-1018) , [new](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf) ]
* **Motivation**
* **Contribution**


### <span id = "ReasoNet">ReasoNet</span>
**ReasoNet: Learning to Stop Reading in Machine Comprehension.** *Yelong Shen, Po-Sen Huang, Jianfeng Gao, and Weizhu Chen.* KDD 2017. [ [pdf](https://arxiv.org/pdf/1609.05284.pdf) ]
* **Motivation**
* **Contribution**


### <span id = "Prediction">Prediction</span>
**Machine Comprehension Using Match-LSTM and Answer Pointer.** *Shuohang Wang, and Jing Jiang.* Arxiv 2016. [[pdf](https://arxiv.org/pdf/1608.07905.pdf)]
* **Motivation**
* **Contribution**


### <span id = "FastQA_Ext">FastQA_Ext</span>
**Making Neural QA as Simple as Possible but not Simpler.** *Dirk Weissenborn and Georg Wiese and Laura Seiffe.* CoNLL 2017. [ [pdf](https://arxiv.org/pdf/1703.04816.pdf) ]
* **Motivation**
* **Contribution**



