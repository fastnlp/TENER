## TENER: Adapting Transformer Encoder for Named Entity Recognition


This is the code for the paper [TENER](https://arxiv.org/abs/1911.04474). 

TENER (Transformer Encoder for Named Entity Recognition) is a Transformer-based model which
 aims to tackle the NER task. Compared with the naive Transformer, we 
 found relative position embedding is quite important in the NER task. Experiments
 in the English and Chinese NER datasets prove the effectiveness.

#### Requirements
This project needs the natural language processing python package 
[fastNLP](https://github.com/fastnlp/fastNLP). You can install by
the following command

```bash
pip install fastNLP
```

#### Run the code

(1) Prepare the English dataset.

##### Conll2003

Your file should like the following (The first token in a line
 is the word, the last token is the NER tag.) 

```
LONDON NNP B-NP B-LOC
1996-08-30 CD I-NP O

West NNP B-NP B-MISC
Indian NNP I-NP I-MISC
all-rounder NN I-NP O
Phil NNP I-NP B-PER

```

##### OntoNotes

Suggest to use the following code to prepare your data 
[OntoNotes-5.0-NER](https://github.com/yhcc/OntoNotes-5.0-NER). 
Or you can prepare data like the Conll2003 style, and then replace the 
OntoNotesNERPipe with Conll2003NERPipe in the code.

For English datasets, we use the Glove 100d pretrained embedding. FastNLP will
 download it automatically.
 
You can use the following code to run (make sure you have changed the 
data path)

```
python train_tener_en.py --dataset conll2003
```
or 
```
python train_tener_en.py --dataset en-ontonotes
```

Although we tried hard to make sure you can reproduce our results, 
the results may still disappoint you. This is usually caused by 
the best dev performance does not correlate well with the test performance
. Several runs should be helpful. 

The ELMo version (FastNLP will download ELMo weights automatically, you just need
to change the data path in train_elmo_en.)

```
python train_elmo_en.py --dataset en-ontonotes
```

   
   
##### MSRA, OntoNotes4.0, Weibo, Resume
Your data should only have two columns, the first is the character,
 the second is the tag, like the following
```
口 O
腔 O
溃 O
疡 O
加 O
上 O
```

For the Chinese datasets, you can download the pretrained unigram and 
bigram embeddings in [Baidu Cloud](https://pan.baidu.com/s/1pLO6T9D#list/path=%2Fsharelink808087924-1080546002081577%2FNeuralSegmentation&parentPath=%2Fsharelink808087924-1080546002081577).
 Download the 'gigaword_chn.all.a2b.uni.iter50.vec' and 'gigaword_chn.all.a2b.bi.iter50.vec'.
 Then replace the embedding path in train_tener_cn.py
 
You can run the code by the following command

```
python train_tener_cn.py --dataset ontonotes
```






