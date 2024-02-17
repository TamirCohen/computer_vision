#TODO!! fix ROC curves, Why the second score is so bad 😠😠😠 😠😠😠 😠😠😠 😠 😠😠 😠😠😠 😠

## Question 8
Test accuracy corresponding for highest validation is 85.52380952380952

## Question 9
1400 real images VS 700 fake

## Question 18
It is pre-trained on datasets with image and labels,
specifically in the paper it was trained on ImageNet and JFT

## Question 19
The basic building blocks comprise depthwise separable convolution batch normalization and relu.

Depthwise separable convolutions, comprise two layers.
Firstly, there is the depthwise convolution layer where each input channel possesses its dedicated kernel. Following this, there is the pointwise convolution layer, which serves to combine all the channels. This approach surpasses traditional convolution since standard convolution kernels have parameters associated with each input channel.

## Question 21
The input size of fc is 2048

## Question 22
The number of parameters of the Xception module is 22,855,952 according to the paper.
get_nof_params returned exactly the same number
`print(get_nof_params(Xception()))`

## Question 24
We added 272834 which is 1.2%

## Question 27
The test accuracy for the highest validation accuracy that we received for the Xception module is 97.82 (WOW!!)

## Question 29
A saliency map for a specific image indicates which pixels are crucial for classifying the image. This map is derived from the class score by calculating the pixel's derivative.

## Question 30
GradCam is a technique to visualize what feature map contributes the most for the image class classification score.
It does so by derivate the classification score by the feature map pixels (of the last cnn layer), and produces a weighted combination of all the pixel maps

## Bonus
Not using resnet because it should not be to deep.
Lets try to use inception modules, or the xception modules?
Xception should be better the inception v3, so maybe lets try decrease its number of params.
Using xception with depthwise separable convolution. 

Option1:
    Parameter goal less than 2 mil
    Stuff to do:
    1. Reduce number of blocks
    2. reduce number of block sizes,
    3. Reduce dimensions of last fc layer from 2048 to 2 using several linear V 
    4. Lower model internal dims
Option2:
    Make Simple net a bit deeper

SimpleNet has 16901762 params and Xception has 22000000 params, which is close. let s try to use the xception than