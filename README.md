[//]: # (Image References)

[image1]: ./images/key_pts_example.png "Facial Keypoint Detection"

# Facial Keypoint Detection

This project was completed as part of Udacity's Computer Vision Nanodegree program.

## Project Overview

In this project, we combine our knowledge of computer vision techniques and deep learning architectures to build a facial keypoint detection system. Facial keypoints include points around the eyes, nose, and mouth on a face and are used in many applications. These applications include: facial tracking, facial pose recognition, facial filters, and emotion recognition. The completed code should be able to look at any image, detect faces, and predict the locations of facial keypoints on each face; examples of these keypoints are displayed below.

![Facial Keypoint Detection][image1]

The project is broken up into a few main parts in four Python notebooks:

__Notebook 1__ : Loading and Visualizing the Facial Keypoint Data

__Notebook 2__ : Defining and Training a Convolutional Neural Network (CNN) to Predict Facial Keypoints

__Notebook 3__ : Facial Keypoint Detection Using Haar Cascades and your Trained CNN

__Notebook 4__ : Fun Filters and Keypoint Uses

Also included are two Python files that are used in the notebooks:

__data_load.py__ : Defining the dataset class and various data transform classes

__models.py__ : Defining the CNN model(s)


## Project Instructions

All of the code and the training/test data are present in this repository. Before you can get started coding, you'll have to make sure that you have all the libraries and dependencies required to support this project.

### Local Environment Instructions

1. Clone the repository, and navigate to the downloaded folder. This may take a minute or two to clone due to the included image data.
```
git clone https://github.com/mallyagirish/facial-keypoints-detection.git
cd facial-keypoints-detection
```

2. Create (and activate) a new environment (called `cv-ml`, for example) with Python 3.6. If prompted to proceed with the install `(Proceed [y]/n)` type y.

	- __Linux__ or __Mac__: 
	```
	conda create -n cv-ml python=3.6
	source activate cv-ml
	```
	- __Windows__: 
	```
	conda create --name cv-ml python=3.6
	activate cv-ml
	```
	
	At this point your command line should look something like: `(cv-ml) <User>:facial-keypoints-detection <user>$`. The `(cv-ml)` indicates that your environment has been activated, and you can proceed with further package installations.

3. Install PyTorch and torchvision; this should install the latest version of PyTorch.
	
	- __Linux__ or __Mac__: 
	```
	conda install pytorch torchvision -c pytorch 
	```
	- __Windows__: 
	```
	conda install pytorch-cpu -c pytorch
	pip install torchvision
	```

6. Install a few required pip packages, which are specified in the requirements text file (including OpenCV).
```
pip install -r requirements.txt
```


### Data

All of the data you'll need to train a neural network is in the subdirectory `data`. In this folder are the training and test sets of images/keypoint data, and their respective csv files. This will be further explored in Notebook 1: Loading and Visualizing Data.


## Notebooks

1. Navigate back to the repo. (Also, your source environment should still be activated at this point.)
```shell
cd
cd facial-keypoints-detection
```

2. Open the directory of notebooks, using the below command. You'll see all of the project files appear in your local environment; open the first notebook and follow the instructions.
```shell
jupyter notebook
```

3. Once you open any of the project notebooks, make sure you are in the correct `cv-ml` environment by clicking `Kernel > Change Kernel > cv-ml`.


LICENSE: This project is licensed under the terms of the MIT license.
