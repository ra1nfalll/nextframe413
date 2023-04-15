# CSC413 Project: Predict next frames in a video using Neural Nets

## Introduction
This project will work on the next frame prediction problem. Next frame prediction is an unsupervised machine learning task to predict the next image frame given a sequence of past frames as input. This involves recognizing objects spatially in an image and their temporal relationships across past frames, which can be interpreted as the subtasks to the next frame prediction problem.

## Model

### Model Figure
(Flowchart goes here)
 
### Model Parameters

### Model Examples 

## Data

### Source
The data used on this project is from the Moving MNIST dataset, which is used for unsupervised learning. The link to access and download the data is: http://www.cs.toronto.edu/~nitish/unsupervised_video/

### Data Summary
The Moving MNIST dataset contains 10,000 sequences each of length 20. Each sequence shows two handwritten digits moving around, each frame is 64 x 64. Each frame has a black backgorund and the numbers that are floating around are white. It is also important to note that the dataset is grayscale. 

### Data Transformation

### Data Split
The current data split we have is 80% of the data for training, 10% for validation, and 10% for testing.

## Training

### Training Curve

### Hyper parameter Tuning

## Results

### Quantitative Measures

### Quantitative and Qualitative Results

### Justification (IMPORTANT 20pts)

## Ethical Consideration

## Authors



### (ARCHIVE)
### Source: 
The main dataset that we will use for our project is the [KTH Action Dataset](https://www.csc.kth.se/cvap/actions/). This dataset is composed of videos of people performing different actorchvision.datasets.ImageFoldertions. There are 6 different actions: walking, jogging, running, boxing, hand waving, hand clapping. The videos were filmed in 4 different scenarios. This dataset is available for non-commercial use as per the website: “The database is publicly available for non-commercial use.”

Instead since the main repository where the data is contained holds it in raw format, as in videos itself, instead we managed to find another copy of this dataset on [kaggle](https://www.kaggle.com/datasets/logicn/kthextract-to-jpg), which has the data videos already split into frames that are needed by the models to make predictions.

Additionally, in an attempt to make the model more generalizable, we also plan on adding data from a different dataset called [UCF Sport dataset](https://www.crcv.ucf.edu/data/UCF_Sports_Action.php), which contains different activities such as diving, lifting, and skateboarding. To use this dataset we need to cite two papers. And for a different testing approach, if time permits, we will use this dataset just for testing so that we can test how our model is performing with unseen data from a different dataset.

### Data Statistics
Please find below some important statistics about the dataset:
1. Number of videos - 599
    1. Training - Original - ? - After New Split - 360
    2. Validation - Original - ? - After New Split - 120
    3. Testing - Original - ? - After New Split - 119

2. Number of Frames - 289716
    1. Training - Original - 89960 - After New Split - 172478
    2. Validation - Original - 93901 - After New Split - 56393
    3. Testing - Original - 105855 - After New Split - 60845

(To be added more... )


### Some preprocessing
If we look closely the data hasn't been split properly for our needs since, test and validation sets have more videos (frames) compared to the training set. Hence we do some more preprocessing, to get our expected split that 60, 20, 20% for train, test and validation sets respectively. 

So to adjust the split we basically merge all the videos into a single videos folder using some simple python code. After that we make our own splits using the `torchvision.datasets.ImageFolder` and `torch.utils.data.Subset` utilites from pytorch. This leaves us with about 360 videos in training, 120 in validation and 119 in testing sets.


## How to use? (Basically setup requirements -- find a word for this)

### Data Loading
Please follow the following steps before running the main notebook to make sure the data is loaded correctly. We are using kaggle to load the data directly:
Reference: https://www.analyticsvidhya.com/blog/2021/06/how-to-load-kaggle-datasets-directly-into-google-colab/

1. Login to your kaggle account and navigate to your account settings and download an API token, on to your computer as shown in the picture.
![image](https://user-images.githubusercontent.com/43979159/229308234-a91b98a8-b538-436a-a11e-95f46c6d6470.png)

2. Next, upload this API token file on to the notebook by going over here. This is your main secret file, don't let anyone else have access to this.

![image](https://user-images.githubusercontent.com/43979159/229308477-f9388bba-3e9c-4bca-a92f-dfd0caab895d.png)

3. Now run the first cell in the notebook and this should connect to the kaggle database and download our dataset that we will use.






