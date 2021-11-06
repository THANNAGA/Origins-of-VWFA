# Origins-of-VWFA
A simple convnet of the [CorNet](https://github.com/dicarlolab/CORnet) family is trained on ImageNet, then on a word dataset of same size. Analyses ensue.

This is the repository for the November 2021 PNAS article:

Emergence of a compositional neural code for written words: Recycling of a convolutional neural network for reading.

## Contents

Here you will find 4 types of resources:

1. Python scripts to train the model.

		clean_cornets.py	- where the models are defined.
		
		clean_train.py		- which contains the training loop and acts like a main script.
		
		ds2.py 		- which holds dataset classes and various functions related to the extraction and manipualtion of data.
  
2. Datasets of stimuli used for the analyses
	
		bodies.zip
		
		faces.zip
		
		houses.zip
		
		tools.zip
		
		words.zip
		
		false_fonts.zip
		
		infreq_letters.zip
		
		freq_letters.zip
		
		freq_bigrams.zip
		
		freq_quadrigrams.zip
		

3. Pretrained models for all conditions.

		save_illit_z_79_full_nomir.pth.tar		- illiterate network checkpoint after 80 epochs of training
		
		save_lit_bias_z_79_full_nomir.pth.tar		- Biased literate network checkpoint after 50 epochs of ImageNet training then 30 epochs of words + ImageNet
		
		save_lit_no_bias_z_79_full_nomir.pth.tar	- Unbiased literate network checkpoint after 50 epochs on ImageNet training then 30 epochs of words + ImageNet
		
		save_pre_z_49_full_nomir.pth.tar		- network checkpoint after 50 epochs of training on ImageNet
		
4. Python scripts to analyze the model.

		Figure_understanding.py
		
		gradient.py 
		
		invariance.py
		
		lesions.py
		
		performance.py
		
		selectivity.py
		
		
		

All models are simple CNNs, trained in Pytorch.


## Requirements

- Python 3.6+
- PyTorch 0.4.1+
- numpy
- tqdm


## License

GNU GPL 3+
