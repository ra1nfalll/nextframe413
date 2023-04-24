# CSC413 Project: Predict next frames in a video using Neural Nets

## Introduction
This project will work on the next frame prediction problem. Next frame prediction is an unsupervised machine learning task to predict the next image frame given a sequence of past frames as input. This involves recognizing objects spatially in an image and their temporal relationships across past frames, which can be interpreted as the subtasks to the next frame prediction problem. We will input into our model a sequence of frames of a video and then output a sequence of next frames in the video. We used the ConvLSTM cell from https://github.com/sladewinter/ConvLSTM. This cell was then embedded in a U-net transformer between the encoder and decoder to predict the features of the next frame. The U-network is an encoder decoder network, where the encoder is a CNN extracting the sequence of features in the video frames, and a decoder reconstructing a new image frame with the output features of the ConvLSTM cell. To assist with the image reconstruction at the inverse maxpool layers in the decoder, skip connections are also utilized to feed forward the outputs from the encoder at each convolutional layer to the inverse convolution applied in the decoder.

We want to aknowledge that the code for this project was based from https://github.com/sladewinter/ConvLSTM, and also the UNet architecture was based from https://discuss.pytorch.org/t/how-to-get-a-5-dimensional-input-with-a-3d-array-tensor/148257/2.


## Model

### Model Figure
![diagram 003](https://user-images.githubusercontent.com/49618034/233893402-13df3480-b34a-4691-961b-46eab67d3b68.jpeg)

 
### Model Parameters
Let's first discuss the implementation of One ConvLSTM.
This model has the following implementation:
Batch Size = 16, Kernel Size = 3x3, Number of Kernels = 64, Padding = 1x1
| Layer/Feature  | Size at Step or of Feature|
| ------------- | ------------- |
| ConvLSTM  | input = [16, 1, 10, 64, 64]  |
| ↳ Hidden State | feature = [16, 64, 64, 64]  |
| ↳ Cell Input | feature = [16, 64, 64, 64]  |
| ↳ ConvLSTM Cell | input = [16, 1, 64, 64]  |
| ↳ ↳ Conv2d | input = [16, 256, 64, 64]  |
| ↳ ↳ Input Gate | feature = [16, 64, 64, 64]  |
| ↳ ↳ Forget Gate | feature = [16, 64, 64, 64]  |
| ↳ ↳ Cell Output | feature = [16, 64, 64, 64]  |
| ↳ ↳ Output Gate | feature = [16, 64, 64, 64]  |
| ↳ ↳ Hidden State | feature = [16, 64, 64, 64]  |
| ↳ ↳ ReLU | input = [16, 64, 64, 64]  |
| ↳ Unrolled Output | input = [16, 64, 10, 64, 64]  |
| ↳ Conv2d | input = [16, 64, 10, 64, 64]  |
| BatchNorm3d  input = | [16, 64, 10, 64, 64]  |
| Output Cut | input = [16, 1, 64, 64]  |
| Sigmoid | input = [16, 1, 64, 64]   |

The parameters that are being trained are as follows:
| Parameter  | Size | # of Parameters |
| ------------- | ------------- | ------------- |
| W_ci | [64, 64, 64] | 262144 |
| W_co | [64, 64, 64] | 262144 |
| W_cf | [64, 64, 64] | 262144 |
| Conv2d (Cell) Weights | [256, 65, 3, 3] | 149760 |
| Conv2d (Cell) Bias | [256] | 256 |
| Conv2d (Sequential) Weights | [1, 64, 3, 3] | 576 |
| Conv2d (Sequential) Bias | [1] | 1 |
| BatchNorm Weights | [64] | 64 |
| BatchNorm Bias | [64] | 64 |


Now let's take a look at the UNet implementation.
This architecture defines a DoubleConv layer which consists of the following:
Given n input channels and m output channels
| DoubleConv  | I/O Channels |
| ------------- | ------------- |
| Conv3d  | [n -> m]  |
| ReLU  | [m -> m]  |
| Conv3d  | [m -> m]  |
| ReLU  | [m -> m]  |

This model has the following implementation:
| Layer  | Size |
| ------------- | ------------- |
| Input | [16, 1, 8, 64, 64] |
| DoubleConv  | [16, 16, 8, 64, 64] |
| MaxPool | [16, 16, 4, 32, 32] |
| DoubleConv  | [16, 32, 4, 32, 32] |
| MaxPool | [16, 32, 2, 16, 16] |
| DoubleConv  | [16, 64, 2, 16, 16] |
| MaxPool | [16, 64, 1, 8, 8] |
| DoubleConv  | [16, 128, 1, 8, 8] |
| ConvTranspose3d | [16, 64, 2, 16, 16] |
| DoubleConv  | [16, 64, 2, 16, 16] |
| ConvTranspose3d | [16, 32, 4, 32, 32] |
| DoubleConv  | [16, 32, 4, 32, 32] |
| ConvTranspose3d | [16, 16, 8, 64, 64] |
| DoubleConv  | [16, 16, 8, 64, 64] |
| Conv2d | [16, 1, 64, 64] |
| Sigmoid | [16, 1, 64, 64] |

We have a lengthy list of trainable parameters for each convolution:
| Trainable Parameters  | Weights Size | Bias Size | # of Parameters |
| ------------- | ------------- | ------------- | ------------- |
| DoubleConv1 Conv1 | [16, 1, 3, 3, 3] | [16] | 448 |
| DoubleConv1 Conv2 | [16, 16, 3, 3, 3] | [16] | 6928 |
| DoubleConv2 Conv1 | [32, 16, 3, 3, 3] | [32] | 13856 |
| DoubleConv2 Conv2 | [32, 32, 3, 3, 3] | [32] | 27680 |
| DoubleConv3 Conv1 | [64, 32, 3, 3, 3] | [64] | 55360 |
| DoubleConv3 Conv2 | [64, 64, 3, 3, 3] | [64] | 110656 |
| DoubleConv4 Conv1 | [128, 64, 3, 3, 3] | [128] | 221312 |
| DoubleConv4 Conv2 | [128, 128, 3, 3, 3] | [128] | 442494 |
| ConvTranpose1 | [128, 64, 2, 2, 2]| [64] | 65600 |
| DoubleConv5 Conv1 | [64, 128, 3, 3, 3] | [64] | 221312 |
| DoubleConv5 Conv2 | [64, 64, 3, 3, 3] | [64] | 110656 |
| ConvTranpose2 | [64, 32, 2, 2, 2] | [32] | 16416 |
| DoubleConv6 Conv1 | [32, 64, 3, 3, 3] | [32] | 55360 |
| DoubleConv6 Conv2 | [32, 32, 3, 3, 3] | [32] | 27680 |
| ConvTranpose3 | [32, 16, 2, 2, 2] | [16] | 4112 |
| DoubleConv7 Conv1 | [16, 32, 3, 3, 3] | [16] | 13856 |
| DoubleConv7 Conv2 | [16, 16, 3, 3, 3] | [16] | 6928 |
| Conv2d | [1, 16, 1, 1] | [1] | 17 |

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
The UNet + ConvLSTM we trained for 10 epochs, because the model stabilizes after the 6th epoch and there is no considerable improvement in learning after the 10th epoch. In order to make a comparison to this model, which is our main model, we trained both the "One ConvLSTM" and the UNet for 10 epochs as well. For the UNet + ConvLSTM we even trained the model for 150 epochs, which lead to no improvement, no visible feature was learned by the model. When choosing the batch size we first started with batch size of 3, but we noticed that this led to a very slow training process, hence we fixed our batch size at 16. The batch size of 16 improved the training time, but did not penalize the training loss considerably. 

## Results

### Quantitative Measures
We evaluate the performance of the model on the test set using a binary pixel wise cross entropy loss. For each output frame in our test video sequence we compare it to the next 10 target frames using the cross entropy loss of the difference in pixel value for our output sequence and the actual frame. As this is an unsupervised learning problem, we solely measure the quality of our output with loss and omit accuracy, as we do not have target labels to compare against.

Additionally, since we noticed the generated images had black pixel noise obscuring the generated digits, we decided to also measure the loss on the features of the generated image after one convolution was applied compared to the features from the same convolution applied to the target. By optimizing based on both the generated pixel intensities and generated features, we hope this new loss can help the model output images of a higher quality by better navigating the loss landscape.


### Quantitative and Qualitative Results
As mentioned in the previous section, we solely use the loss measure the quantitative performance of our model. Please check below the results for each of the three models we tested against.

| Model            | Training Loss    | Validation Loss      |
| -----------------| ---------------- | ---------------------|
| One ConvLSTM     |       405.35     |         412.47       |
| UNet             |       330.81     |         336.60       |
| UNet + ConvLSTM  |       693.38     |         673.05       |

As for Qualitative Measures, since we are predicting the next frame of videos, one very simple and applicable quality measure is visual inspection. We can clearly see from the output of the three different models from the models section, that the frames generated are not as clear and visible as the expected frames for the One ConvLSTM and UNet architecture. As for the UNet + ConvLSTM structure, we don't see any results generated at all and it results only in a completely black frame.

| Model            | Expected | Result |
| ---------------- | ---------------- | -------------------- |
| One ConvLSTM     | ![aac82066-e22b-4aa8-918a-616f3d761cf6](https://user-images.githubusercontent.com/49618034/233894318-afeff0a8-d84f-442d-a2ca-fed01bfb2bcf.gif) | ![6be549c2-0f57-4fc4-b42b-3dbdcfcec462](https://user-images.githubusercontent.com/49618034/233894368-86b90cbb-f63b-4946-9c60-d68de6d4a2f7.gif) |
| UNet             | ![dd47a1ab-2e9e-4353-9cbd-9521c98952a3](https://user-images.githubusercontent.com/49618034/233894475-1a1647e0-a38c-4f23-81ce-362e18c8ce55.gif) | ![2eca8664-7e26-4279-93f1-9078c6b606e1](https://user-images.githubusercontent.com/49618034/233894509-8b3ead6b-f8bf-47e7-b8e9-4f2daacbc673.gif) |
| UNet + ConvLSTM  | ![9d0a46f6-6c30-4df8-88f0-95a8518504c7](https://user-images.githubusercontent.com/49618034/233894573-a5dab0c1-1dc7-4380-9a35-e18145f00163.gif) | ![e22ecd5a-4545-4cd0-b0b9-aa46393261d6](https://user-images.githubusercontent.com/49618034/233894727-93ef95ce-00c9-432b-b939-b2d8af9d3018.gif) |

### Justification (IMPORTANT 20pts)
We believe our model performed reasonably well. The task we originally set up to accomplish was to predict the next frame of a video from one of 2 different datasets by the name of UCFSports and KTH. UCFSports contained 150 videos at 10 frames and a 720x480p resolution while KTH contained 600 videos at 25 frame and a 160x120p resolution. The videos showed different human actions such as running, walking or diving. The size of the dataset and length of training time required for simply anything made us abandon the idea and revert back to the current used dataset; Moving MNIST. MovingMNIST is a much simpler dataset, containing 10000 videos of 20 frames at 64x64p resolution. A much smaller resolution enabled us to train on and test with this dataset easier. MovingMNIST is simply videos of handdrawn digits from the famous MNIST dataset bouncing around a static black background. It is essentially a default dataset for this type of problem. We unfortunately did not attain the level of loss we would have liked to reach during the course of training and testing our model. 

Our loss for our best model, which is the one containing UNet reached just below 350. We were aiming for a target of under 300. We did get results that appear with the human eye to look appropriate for the next frame. [EXPLANATION HERE OF HOW OUR QUANTITATIVE AND QUALITATIVE RESULTS SEEM]. We feel we did extensive testing with hyperparameters and different models throughout our time working. Many attempts at models are not shown here because they simply didn't pan our for a variety of reasons. Some things that occured include a too lengthy training time or the model not training well. Our main hyperparameters in batch size and epochs to run for were found through many tests. Other parameters such as number of layers, input/output features, padding, activations etc. required some tinkering but we were able to use a basis in the models that we sampled from for many of these. 

We found that the issue with a lot of our original attempts at models revolved around too many layers and complicated models where we tried to combine different architectures. We ran into a lot of difficulties with fitting sizes together and ensuring training progressed smoothly. When we tried working with simpler concepts such as a single ConvLSTM layer and adapting the UNet architecture we were able to better tune the model and understand the architecture. This led to developing well-tuned models that gave us decent results. Overall, we managed to get decent results in the end, but not as great as a result as we might have wanted. The output frames for example are still a little fuzzy and the digits not very well defined. Nevertheless, we are pleased with our current results but feel they could be better if developed further.

## Ethical Consideration
1. Our model is used to generate frames for a video. If you consider this on a larger scale than the dataset that we are using then there may be issues with video prediction. If a large enough network could be trained on a large portion of YouTube for example, anyone in those videos could have frames of a video generated from them. This could be a huge privacy issue for people. With this current model specifically trained on the Moving MNIST dataset we would probably not have much issue. 
2. Bias in datasets (bias in subject or actions): images generated may be better at representing certain groups of people or behaviors over represented in the dataset, or associate certain physical features of population groups to temporal actions from a bias in the data.
3. In robotic decision making and autonomous driving: ethics of decision making based on the model, as biases in the model will be inherited by the decision making algorithms (Ex: if the model cannot predict future actions of racial minorities accurately, a decision made based upon output for autonomous driving or robotic applications can be dangerous.
4. Self-driving cars accident prevention: Self-driving cars have both sensors and cameras, using our model it would be possible for the car to predict the future positions of objects in a scene to avoid accidents.
5. Military Use: Since this model predicts where objects may be in the future, it could be used for predicting positions of moving targets.


## Authors
| Student               | Worked On                                  |
| --------------------- | ------------------------------------------ |
| Thiago | Developing and training the model, fixing bugs, plotting training curves, drawing the model, augmenting the data, gathering examples of expected and predicted results, organizing the colab files, and writing the README. |
| Shrey  | Researching different model architechtures, looking into initial datasets such as the KTH dataset and organizing it. Applying the data augmentation to invert the image pixels. Troubleshooting the models to have them generate better results, and writing the README. | 
| Ryan | Looking into inital datasets, organizing initial tasks, training the model, fixing bugs, collecting and analyzing data, beta model development, model parameter analysis, and writing the README. |









