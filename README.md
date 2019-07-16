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


dataprocessing
------
### used in IBM model 1

tensorflow_and_keras
------
### used in IBM model 1

BERT
------
### used in IBM model 1
