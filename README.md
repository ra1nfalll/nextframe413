# CSC413 Project: Predict next frames in a video using Neural Nets

## Introduction
This project will work on the next frame prediction problem. Next frame prediction is an unsupervised machine learning task to predict the next image frame given a sequence of past frames as input. This involves recognizing objects spatially in an image and their temporal relationships across past frames, which can be interpreted as the subtasks to the next frame prediction problem. We will input into our model a sequence of frames of a video and then output a sequence of next frames in the video. We used the ConvLSTM cell from https://github.com/sladewinter/ConvLSTM. This cell was then embedded in a U-net transformer between the encoder and decoder to predict the features of the next frame. The U-network is an encoder decoder network, where the encoder is a CNN extracting the sequence of features in the video frames, and a decoder reconstructing a new image frame with the output features of the ConvLSTM cell. To assist with the image reconstruction at the inverse maxpool layers in the decoder, skip connections are also utilized to feed forward the outputs from the encoder at each convolutional layer to the inverse convolution applied in the decoder.

We want to aknowledge that the code for this project was based from https://github.com/sladewinter/ConvLSTM, and also the UNet architecture was based from https://discuss.pytorch.org/t/how-to-get-a-5-dimensional-input-with-a-3d-array-tensor/148257/2.


## Model

### Model Figure
![diagram 003](https://user-images.githubusercontent.com/49618034/233893402-13df3480-b34a-4691-961b-46eab67d3b68.jpeg)

 
### Model Parameters
(NEED TO BE FILLED)

### Model Examples 
We were not able to have a successful result using the UNet combined with the ConvLSTM, so below we will present two successful examples of adapatations of the original code, one using only one ConvLSTM to make predictions, and the other just the UNet to make predictions. The unsuccessful result will be from the UNet combined with the ConvLSTM as showed in the diagram above. 

| Model            | Expected | Result |
| ---------------- | ---------------- | -------------------- |
| One ConvLSTM     | ![aac82066-e22b-4aa8-918a-616f3d761cf6](https://user-images.githubusercontent.com/49618034/233894318-afeff0a8-d84f-442d-a2ca-fed01bfb2bcf.gif) | ![6be549c2-0f57-4fc4-b42b-3dbdcfcec462](https://user-images.githubusercontent.com/49618034/233894368-86b90cbb-f63b-4946-9c60-d68de6d4a2f7.gif) |
| UNet             | ![dd47a1ab-2e9e-4353-9cbd-9521c98952a3](https://user-images.githubusercontent.com/49618034/233894475-1a1647e0-a38c-4f23-81ce-362e18c8ce55.gif) | ![2eca8664-7e26-4279-93f1-9078c6b606e1](https://user-images.githubusercontent.com/49618034/233894509-8b3ead6b-f8bf-47e7-b8e9-4f2daacbc673.gif) |
| UNet + ConvLSTM  | ![9d0a46f6-6c30-4df8-88f0-95a8518504c7](https://user-images.githubusercontent.com/49618034/233894573-a5dab0c1-1dc7-4380-9a35-e18145f00163.gif) | ![e22ecd5a-4545-4cd0-b0b9-aa46393261d6](https://user-images.githubusercontent.com/49618034/233894727-93ef95ce-00c9-432b-b939-b2d8af9d3018.gif) |

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
In order to increase the number of videos that we have to train our model, we inverted the colors of every single video, generating an augmented dataset with double the size of the original one. The process to achieve this result is: First we need to loop through every video in the original dataset, and subsquently loop over each frame. Then we used a pytorch function "torchvision.transforms.functional.invert" to invert the grayscale values of each frame, and finally we combined all the frames and added both the original video and the new video to a augmented dataset. Below there are a few examples of what the new examples look like:

![468357a8-4e61-45f0-a2d0-91fcf8ea098e](https://user-images.githubusercontent.com/49618034/232344696-8db1c365-e490-4449-a0c1-5d14234d80d5.gif)
![58639a2d-6db2-49a3-9566-96c3f91bb446](https://user-images.githubusercontent.com/49618034/232344698-e48bc068-65df-4566-b809-ca24f0bcb048.gif)
![b5a75827-2389-419a-b03f-98a0606e0c24](https://user-images.githubusercontent.com/49618034/232344699-8d2c03e5-b247-498b-b224-264e6bbf9036.gif)



### Data Split
The data below was the desired data split to train the model, but since we were not getting any features learned by the model (UNet + ConvLSTM) we chose to run with less data, since more data did not improve the performance, it only slowed the training process. 
Desired split:
| Split      | Number of Videos | Percentage of Videos |
| ---------- | ---------------- | -------------------- |
| Training   |       8000       |         80%          |
| Validation |       1000       |         10%          |
| Test       |       1000       |         10%          |

Actual split:
| Split      | Number of Videos | Percentage of Videos |
| ---------- | ---------------- | -------------------- |
| Training   |       2000       |         20%          |
| Validation |       256       |         2.56%          |
| Test       |       256       |         2.56%          |


## Training

### Training Curve
| Model            | Loss Curve       | 
| ---------------- | ---------------- | 
| One ConvLSTM     |  ![download](https://user-images.githubusercontent.com/49618034/233895057-5c1a82d8-bcf8-4334-9fd3-1896c0839564.png) |
| UNet             |  ![download](https://user-images.githubusercontent.com/49618034/233895099-18ac3f5a-8d6f-4cdd-b01e-7bdf3307bd41.png) | 
| UNet + ConvLSTM  |  ![download](https://user-images.githubusercontent.com/49618034/233895145-df364043-cf2c-42f0-a2f5-23276d21c486.png) |

### Hyper parameter Tuning
The UNet architecture led to a slower learning process compared to the original model. Therefore, for the UNet architecture we chose to run for 10 epochs, and then we noticed that training for a greater number of epochs did not lead to big improvements in the loss of the model. When choosing the batch sizes we first experimented with lower batch sizes such as 3, and then we started to increase it in our different iterations of training the model. The biggest batch size we reached was 30, and we noticed that the batch that led to the faster progress while training was the batch size of 16. 

## Results

### Quantitative Measures
We evaluate the performance of the model on the test set using a binary pixel wise cross entropy loss. For each output frame in our test video sequence we compare it to the next 10 target frames using the cross entropy loss of the difference in pixel value for our output sequence and the actual frame. As this is an unsupervised learning problem, we solely measure the quality of our output with loss and omit accuracy, as we do not have target labels to compare against.

Additionally, since we noticed the generated images had black pixel noise obscuring the generated digits, we decided to also measure the loss on the features of the generated image after one convolution was applied compared to the features from the same convolution applied to the target. By optimizing based on both the generated pixel intensities and generated features, we hope this new loss can help the model output images of a higher quality by better navigating the loss landscape.



### Quantitative and Qualitative Results
(NEEDS TO BE FILLED)

### Justification (IMPORTANT 20pts)
(NEEDS TO BE FILLED)

## Ethical Consideration
Our model is used to generate frames for a video. If you consider this on a larger scale than the dataset that we are using then there may be issues with video prediction. If a large enough network could be trained on a large portion of YouTube for example, anyone in those videos could have frames of a video generated from them. This could be a huge privacy issue for people. With this current model specifically trained on the Moving MNIST dataset we would probably not have much issue.


## Authors
| Student               | Worked On                                  |
| --------------------- | ------------------------------------------ |
| Thiago | Developing and training the model, fixing bugs, plotting training curves, drawing the model, augmenting the data, gathering examples of expected and predicted results, organizing the colab files, and writing the README. |









