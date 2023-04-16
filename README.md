# CSC413 Project: Predict next frames in a video using Neural Nets

## Introduction
This project will work on the next frame prediction problem. Next frame prediction is an unsupervised machine learning task to predict the next image frame given a sequence of past frames as input. This involves recognizing objects spatially in an image and their temporal relationships across past frames, which can be interpreted as the subtasks to the next frame prediction problem. We will input into our model a sequence of frames of a video and then output a sequence of next frames in the video.


## Model

### Model Figure
(Flowchart goes here)
 
### Model Parameters
Our model uses 3 convolutional LSTM networks followed by a final convolutional 2d layer. Each one is separated by a batch normalization layer.

### Model Examples 

## Data

### Source
The data used on this project is from the Moving MNIST dataset, which is used for unsupervised learning. It is sourced directly from the University of Toronto from a paper published in 2015 by Nitish Srivastava, Elman Mansimov, and Ruslan Salakhutdinov.The link to access and download the data is: http://www.cs.toronto.edu/~nitish/unsupervised_video/

### Data Summary
The Moving MNIST dataset contains 10,000 sequences each of length 20. Each sequence shows two handwritten digits moving around, each frame is 64 x 64. Each frame has a black backgorund and the digits that are floating around are white. The digits bounce off the edges of the frame and continue moving appropriately after bouncing, but they do not interact with each other and do not bounce off each other. Digits instead simply overlap and continue moving on their current trajectory with no regard for the other digit. It is also important to note that the dataset is grayscale. Below we have a few example of the images, and also GIFs showing the video.
<img width="1197" alt="Screen Shot 2023-04-16 at 1 40 33 AM" src="https://user-images.githubusercontent.com/49618034/232273410-108cebd9-6d18-46cf-a2c5-5b4c973fcdce.png">
![f5a50978-13bf-4609-88e5-50f566667934](https://user-images.githubusercontent.com/49618034/232273460-6fd7d6e3-8706-42f3-a8a4-d648f2e5af85.gif)
![7462818e-3311-416c-9d8d-bbb2fa39fc3e](https://user-images.githubusercontent.com/49618034/232273463-29297307-0b33-4ca5-8ffc-5abbbbdda78f.gif)
![94e461ba-cee2-4131-a943-0756c48f8887](https://user-images.githubusercontent.com/49618034/232273467-32942af9-5a13-4042-92f0-61d980e5b93a.gif)


### Data Transformation
We took the data directly from the website using the command wget where it is already inthe form of a numpy array and then load it in directly. In order to use the data in our neural network we performed a few small transformations after loading it in. First, we unsqueezed the data to add a channel dimension because currently each video frame is of the dimensions 64x64x1 and PyTorch models expect a 4 dimensional tensor so we add an additional dimension at the front that represents our number of input channels of 1. Secondly since each pixel in a frame holds a grayscale value from 0 to 255, we normalize the data to be within 0 and 1 to help us stabilize the gradient descent algorithm.


### Data Split
| Split      | Number of Videos | Percentage of Videos |
| ---------- | ---------------- | -------------------- |
| Training   |       8000       |         80%          |
| Validation |       1000       |         10%          |
| Test       |       1000       |         10%          |


## Training

### Training Curve

### Hyper parameter Tuning

## Results

### Quantitative Measures
We evaluate the results using a binary pixel wise cross entropy loss. For each outputted frame in our sequence we compare it to the next 10 actual frames. Then we compare the cross entropy loss of the difference in pixel value for our outputted sequence and the actual.


### Quantitative and Qualitative Results

### Justification (IMPORTANT 20pts)

## Ethical Consideration
Our model is used to generate frames for a video. If you consider this on a larger scale than the dataset that we are using then there may be issues with video prediction. If a large enough network could be trained on a large portion of YouTube for example, anyone in those videos could have frames of a video generated from them. This could be a huge privacy issue for people. With this current model specifically trained on the Moving MNIST dataset we would probably not have much issue.


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






