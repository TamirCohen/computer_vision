## Question 8
Test accuracy corresponding for highest validation is 85.52380952380952

## Question 9
1400 real images VS 700 fake

## Question 18
It is pre-trained on datasets with image and labels,
specifically in the paper it was trained on ImageNet and JFT

## Question 19
The basic building blocks comprise depthwise seperable convolution batch normalization and relu.

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