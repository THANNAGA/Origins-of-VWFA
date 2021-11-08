# Origins-of-VWFA

This is the repository for the November 2021 PNAS article:

[Emergence of a compositional neural code for written words: Recycling of a convolutional neural network for reading](https://www.pnas.org/content/118/46/e2104779118)<br>
T. Hannagan*, A. Agrawal*, L. Cohen, S. Dehaene<br>
Proceedings of the National Academy of Sciences Nov 2021, 118 (46) e2104779118; DOI: 10.1073/pnas.2104779118<br>

A simple convnet of the [CorNet](https://github.com/dicarlolab/CORnet) family is trained on ImageNet, then on a word dataset of same size. Analyses ensue.

## Contents

Here you will find 4 types of resources:

1. Python scripts to train the model.

		clean_cornets.py	- where the models are defined.
		
		clean_train.py		- contains the training loop and acts like a main script.
		
		ds2.py 		- holds dataset classes and various functions related to the extraction and manipualtion of data.
		
		Note: Training the model requires downloading the ImageNet dataset, and either obtaining the additional word dataset from the first author (the dataset exceeds the size limit provided by Github), or generating word stimuli anew. The code for generating word stimuli can be found in the script "ds2.py".
		
2. Pretrained models for all conditions.

		save_illit_z_79_full_nomir.pth.tar		- illiterate network checkpoint after 80 epochs of training
		
		save_lit_bias_z_79_full_nomir.pth.tar		- Biased literate network checkpoint after 50 epochs of ImageNet training then 30 epochs of words + ImageNet
		
		save_lit_no_bias_z_79_full_nomir.pth.tar	- Unbiased literate network checkpoint after 50 epochs on ImageNet training then 30 epochs of words + ImageNet
		
		save_pre_z_49_full_nomir.pth.tar		- network checkpoint after 50 epochs of training on ImageNet
		
3. Datasets of stimuli used for the analyses
	
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
