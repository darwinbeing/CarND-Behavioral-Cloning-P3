# **Behavioral Cloning**

![alt text][image1]

---

## Behavioral Cloning Project

### Introduction

In this project, we architect and train a deep neural network to drive a car in a simulator. Collect my own training data and use it to clone my own driving behavior on a test track.


The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)
[image1]: ./resources/preface.png "Preface"
[image2]: ./resources/sheet1.png "Sheet 1"
[image3]: ./resources/left.jpg "Left Image"
[image4]: ./resources/center.jpg "Center Image"
[image5]: ./resources/right.jpg "Right Image"
[image6]: ./resources/histogram1.png "Histogram 1"
[image7]: ./resources/histogram2.png "Histogram 2"
[image8]: ./resources/preprocessed.png "Preprocessed Image"
[image9]: ./resources/augmented.png "Augmented Image"
[image10]: ./resources/mse_loss.png "Mean Squared Error Loss"


### Project Resources

we need driving data and simulator to complete the Behavioral Cloning Project.

* [Sample Training Data](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip).
* [Simulator - macOS](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4594_mac-sim.app/mac-sim.app.zip)

The sample training data provided by Udacity only include sample driving data for the first track.
The simulator contains two tracks, we may need to collect additional data in order to get the vehicle to stay on the track two.
The simulator contains training mode and autonomous mode, one is to record images and steering angles while driving the car around the tracks, the other is to train network.
I am running the driving simulator on my MacBook Pro (Retina, 15-inch, Mid 2015) with 800x600 resolution and fantastic quality.

### Data Set Summary & Exploration

**Directory structure**

```
├── track1
│   ├── IMG
│   └── driving_log.csv
├── track2
│   ├── IMG
│   └── driving_log.csv
└── udacity
    ├── IMG
    └── driving_log.csv
```

* IMG folder - this folder contains all the frames of your driving.
* driving_log.csv - each row in this sheet correlates your image with the steering angle, throttle, brake, and speed of your car. You'll mainly be using the steering angle.

My data includes:

* track 1: Udacity data.
* track 1: 2 laps manually created, driving in the center, one lap driving clock wise, on lap driving counter-clockwise.
* track 2: 5 laps manually created, three laps of center lane driving, one lap of recovery driving from the sides, one lap focusing on driving smoothly around curves.


**Summary dataset**

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 16101
* The size of the validation set is 4026
* The shape of a image is (160, 320, 3)

**Data Visualization**

![alt text][image2]

The figure above is driving log csv file of the udacity data, which contains header and relative path of the image, while the data manually created does not have a header with an absolute path, we need to take into accout when loading data.

![alt text][image3] ![alt text][image4]![alt text][image5]

The images above are taken from left/center/right camera of the simulator, we mainly use the images and steering angle to train my network.

Here is an exploratory visualization of the training dataset of udacity.

![alt text][image6]

From the figure above, the data is greatly imbalanced, with an overwhelming number of steering wheel data being zero, which means, that unless we take corrective steps, so the model will have bias associated with going straight.

![alt text][image7]

I drop something like 90% of the data that has angles near to zero, but not all data point with a steering angle of 0, but this wasn't used in the end.

```
zero_df = df[driving_log['steering'] == 0]
zero_df = zero_df.sample(frac=0.9, random_state=42)
df = df[~df.index.isin(zero_df.index)]
```



### Model Architecture and Training Strategy

**Data preprocessing**

The steps I take in preprocessing data is to crop the original image, 70 rows pixels from the top of the image, 20 rows pixels from the bottom of the image, resize to 64x224, split into YUV planes and passed to the network.

![alt text][image8]


**Data augmentation**

The data augmentation I used was to brighten, darken the images, add artificial shifts and rotations and flip them left to right randomly.

![alt text][image9]

**Dataset split**

I shuffle and split the data into training(80%) and validation(10%) datasets.

**Using multiple cameras**

I use left/right camera image with a correction factor of 0.2 as network input, becase this will increase the dataset and help teach the network steer back to the center if the vehicle starts drifting off to the side.


```
 choice = np.random.choice(3)
 image = mpimg.imread(X[index][choice])
 steering_angle = y[index] + steering_angle_cal[choice]

```

**Generator**

Generators can be a great way to work with large amounts of data. Instead of storing the preprocessed data in memory all at once, using a generator you can pull pieces of the data and process them on the fly only when you need them, which is much more memory-efficient.
I shuffle data in generator, because the data comes from a video sequence so we need to shuffle it in order to generalize the model, otherwise, each epoch would have exactly the same batch data.

**Network Architecture**

This project was inspired by the Nvidia paper [End To End Learning For Self Driving Cars](https://arxiv.org/abs/1604.07316)

The network consists of 9 layers, including a normalization layer, 5 convolutional layers and 3 fully connected layers.

The first layer of the network performs image normalization. The normalizer is hard-coded and is not adjusted in the learning process. Performing normalization in the network allows the normalization scheme to be altered with the network architecture, and to be accelerated via GPU processing.

The convolutional layers are designed to perform feature extraction, and are chosen empirically through a series of experiments that vary layer configurations. We then use strided convolutions in the first three convolutional layers with a 2×2 stride and a 5×5 kernel, and a non-strided convolution with a 3×3 kernel size in the final two convolutional layers.

We follow the five convolutional layers with three fully connected layers, leading to a final output control value which is the inverse-turning-radius. The fully connected layers are designed to function as a controller for steering, but we noted that by training the system end-to-end, it is not possible to make a clean break between which parts of the network function primarily as feature extractor, and which serve as controller.


My final model consisted of the following layers:

<pre>
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lambda_1 (Lambda)            (None, 64, 224, 3)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 30, 110, 24)       1824
_________________________________________________________________
batch_normalization_1 (Batch (None, 30, 110, 24)       96
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 13, 53, 36)        21636
_________________________________________________________________
batch_normalization_2 (Batch (None, 13, 53, 36)        144
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 25, 48)         43248
_________________________________________________________________
batch_normalization_3 (Batch (None, 5, 25, 48)         192
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 23, 64)         27712
_________________________________________________________________
batch_normalization_4 (Batch (None, 3, 23, 64)         256
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 21, 64)         36928
_________________________________________________________________
batch_normalization_5 (Batch (None, 1, 21, 64)         256
_________________________________________________________________
flatten_1 (Flatten)          (None, 1344)              0
_________________________________________________________________
dense_1 (Dense)              (None, 100)               134500
_________________________________________________________________
batch_normalization_6 (Batch (None, 100)               400
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050
_________________________________________________________________
batch_normalization_7 (Batch (None, 50)                200
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510
_________________________________________________________________
batch_normalization_8 (Batch (None, 10)                40
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11
=================================================================
Total params: 273,003
Trainable params: 272,211
Non-trainable params: 792
_________________________________________________________________
</pre>

I modified the Nvidia model by adding batch normalization after the non-linearity of the current layer, which fulfills some of the same goals as Dropout. Removing Dropout from Modified BN-Inception speeds up training, without increasing overfitting and enables higher learning rates.

As optimizer the Adam Optimizer with a learning rate of 0.001.

Here are the parameters:

* EPOCHS: 200
* BATCH_SIZE: 32
* learning rate: 0.001
* test size = 0.2

I can get the car driving on the track 1 with modified Nvidia model and Udacity data, but fell off track 2.

To increase robustness to camera shifts and rotations, slight random translations and rotations to the original image were used as data augmentation, similar to the NVIDIA paper. This did not seem to improve performance significantly.
Instead of modifying the model, I decide to collect more data.

**Data collection**


To acquire training data I used an Xbox one controller to collect more data.
For track 1, driving in the center, one lap driving clock wise, on lap driving counter-clockwise.
For track 2, driving 3 laps and tried to keep in the center of the track. one lap of recovery driving from the sides, one lap focusing on driving smoothly around curves.


**Visualizing Loss**

Keras outputs a history object that contains the training and validation loss for each epoch. we use the history object to visualize the loss, the figure shows that training for up to 115 epochs would result in better performance

![alt text][image10]

### Video


**Track 1**

[![Track 1](https://img.youtube.com/vi/LOVovqoxU6A/0.jpg)](https://www.youtube.com/watch?v=LOVovqoxU6A "Track 1")

**Track 2**

[![Track 2](https://img.youtube.com/vi/LtTZwTHH9Fo/0.jpg)](https://www.youtube.com/watch?v=LtTZwTHH9Fo "Track 2")

### Conclusions
* The CNNs are able to learn the entire task of lane and road
following without manual decomposition into road or lane marking detection, semantic abstraction,
path planning, and control.
* High quality driving data is very important.
