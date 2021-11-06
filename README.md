# Origins-of-VWFA
A convnet of the CorNet family is trained on ImageNet, then on a word dataset of same size. Analyses ensue.

This is the repository for the November 2021 PNAS article:
Emergence of a compositional neural code for written words: Recycling of a convolutional neural network for reading.

Here you will find 3 types of resources:
1. Python scripts to train the model
	These consist in 3 scripts:	
  	"clean_cornets.py" where the models are defined.\
  	"clean_train.py" which contains the training loop and acts like a main script.
  	"ds2.py" which holds dataset classes and various functions related to the extraction and manipualtion of data.
  
2. Datasets of stimuli used for the analyses
		

3. Pretrained models for all conditions.

All models are very simple CNNs, trained in Pytorch.
