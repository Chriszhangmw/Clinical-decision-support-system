Clinical-decision-support-system
===========================
Clinical-decision-support-system

****
	
|Author|Chris Zhang|
|---|---
|E-mail|zhangmw_play@163.com


****
## content
* [dataanalysis](#dataanalysis)
* [dataprocessing](#dataprocessing)
* [tensorflow_and_keras](#tensorflow_and_keras)
* [BERT](#BERT)

dataanalysis
-----------
### introducation:
1. The goal of the demo is to build a clinical diagnosis prediction model which can be used by Hospital to identify patients with diseases during early diagnosis stage.The clinical diagnosis prediction model will be built on historical patients’ records, and later can be used to predict illness base on new patients’ conditions.
2. the patients records contains Chief Complaint, History of Current Illness and History of Past illness, so the first step is to transfor the words to vector and input with model
3. the goal of building model is to create 14 binary classifiers, 8 for body part and 6 for symptom， and the whole 14 classifiers results can be referenced by doctor

### distribution:
1. Attribute:
![1](https://raw.github.com/Chriszhangmw/Clinical-decision-support-system/master/picture/datatype.png)
2. age and gender distribution：
![2](https://raw.github.com/Chriszhangmw/Clinical-decision-support-system/master/picture/age.png)
3. label :
![3](https://raw.github.com/Chriszhangmw/Clinical-decision-support-system/master/picture/labelwithage.png)
![4](https://raw.github.com/Chriszhangmw/Clinical-decision-support-system/master/picture/label.png)
4. body part:
![5](https://raw.github.com/Chriszhangmw/Clinical-decision-support-system/master/picture/bodypartnegative.png)
![6](https://raw.github.com/Chriszhangmw/Clinical-decision-support-system/master/picture/bodypart.png)
![7](https://raw.github.com/Chriszhangmw/Clinical-decision-support-system/master/picture/bodypartexcel.png)
5.symptom:
![8](https://raw.github.com/Chriszhangmw/Clinical-decision-support-system/master/picture/symptomnagetive.png)
![9](https://raw.github.com/Chriszhangmw/Clinical-decision-support-system/master/picture/symbol.png)
![10](https://raw.github.com/Chriszhangmw/Clinical-decision-support-system/master/picture/symptonexcel.png)
6. the sentence length distribution:
![11](https://raw.github.com/Chriszhangmw/Clinical-decision-support-system/master/picture/sequencelength.jpg)

dataprocessing
------
Obviously, the distribution of positive and negative samples is extremely unbalance，so，the first step we should solve the unbalanced data, there are two options：undersample and oversample， oversample  based on the final inputs, but our sample will under NLP(make sample into vectors) before input in the neural networks, so , it is  difficult for us to do oversample,  and I choose undersample， but if we do undersample based on 922 samples, the negative samples only 70 in total, how about we do data enhancement before undersample. 
Let’s look the sample in detail:
### data enhancement:
	1.Chief complaint: X；Current medical history：Y；Past history：Z
	2.Whether using Word embedding or character embedding，Before we do vectorization, we must convert each word into its own index number, so changing the order of the three parts of the statement will make the result of the vectorization completely different. I call these step as data enhancement
	3.+1,Chief complaint: X；Past history：Z；Current medical history：Y
	4.+1,Current medical history：Y；Chief complaint: X；Past history：Z
	5.+1,Current medical history：Y；Past history：Z；Chief complaint: X
	6.+1,Past history：Z；Chief complaint: X；Current medical history：Y
	7.+1,Past history：Z；Current medical history：Y；Chief complaint: X
and now ,after data enhancement, we have 922*6 samples in total, then, we can do undersample.

tensorflow_and_keras
------
1.Embedding methods：
	1.Rand: Randomly initialize the word vector of each word  and adjust through the later  training
	2.For example: character embedding （TensorFlow）,embedding layer （Keras）                        
	3.Static：Use pre-trained word vectors, such as word vectors trained by word2vec, will not adjust the word vector during training
	4.For example: Word embedding（gensim）
	5.Genism：Word---index---vector（trained）
	6.Sklearn：Word---index---tf-idf（fixed）
	7.Keras :  Word ---- index----vector（random）
### design CNN + LSTM:
![12](https://raw.github.com/Chriszhangmw/Clinical-decision-support-system/master/picture/design.png)
results:
![13](https://raw.github.com/Chriszhangmw/Clinical-decision-support-system/master/picture/liverresult.png)
![14](https://raw.github.com/Chriszhangmw/Clinical-decision-support-system/master/picture/stomachresult.png)
### design textCNN:
The CNN filter’s width should be the same as vector’s dimension，however the height can be  as 3,4,5 , I think the most important point is because each row of vector we input represents a word, and in the process of nature language process, the word is the minimum granularity of the text. Compared computer version， pixel  is is the minimum granularity of the picture
results:
![15](https://raw.github.com/Chriszhangmw/Clinical-decision-support-system/master/picture/textacc.png)
![16](https://raw.github.com/Chriszhangmw/Clinical-decision-support-system/master/picture/textloss.png)
![17](https://raw.github.com/Chriszhangmw/Clinical-decision-support-system/master/picture/textConfusion.png)
             


BERT
------
### used in IBM model 1
