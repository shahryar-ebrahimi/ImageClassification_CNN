# ImageClassification_CNN
A Convolutional Neural Network was used to classify images in CIFAR-10 dataset.



P_1:

we are trying to implement a neural network suitable for
classifying of the CIFAR-10 dataset. To do this, we implement a neural network with these
feature that have been mentioned below:
We used a neural network including 3 convolutional layers and one fully-connected layer.
We have implemented just one 2D convolution in each convolutional layer.
Features of neural network:
The first layer consists one 2D Convolution.
The second layer consists one 2D Convolution.
The third layer consists one 2D Convolution.

 The fully-connected layer:
   The first layer:
     Number of neurons = 512
     Activation Function = RELU
   The second layer:
     Number of neurons = 10
     Activation function = Softmax


In our implementation, the size of mini-batch is 32 according to the standard, and we
have used “RMSProp” as an optimizer. The “Cross-entropy” has been used as loss function.
In this problem, we have been asked to plot the graph of the loss and accuracy for both
train and test datasets. They have been showed in below separately.

The best accuracy for train data after 50 epochs is about 99%, and the best loss is about
0.07.
The best accuracy for test dataset almost after 50 epochs is about 71%, and the best loss
almost after 15 epochs is about 0.8. Therefore, we can stop the training after almost 15
epochs, because the loss for test data is not decreased.


P_2:

we want to evaluate the neural network with three different activation
functions. We are trying to compare sigmoid, rectified linear unit, and exponential linear
unit. The activation functions have been compared in such a way that the number of was
10.

The “Sigmoid” gives rise to a problem of “vanishing gradients”, but RELU avoids and
rectifies vanishing gradient problem. Also, RELU is less computationally expensive than
sigmoid because it involves simpler mathematical operations. As you can see in figure.2-1
to figure.2-4 the RELU activation function has better performance compared to sigmoid.
ELU is very similar to RELU except negative inputs. They are both in identity function
form for non-negative inputs. On the other hand, ELU becomes smooth slowly until its
output equal to -α whereas RELU smoothes sharply. ELU is a strong alternative to RELU.
Unlike to RELU, ELU can produce negative outputs. According to results, it is obvious both
RELU and ELU are similar but ELU is a little better than RELU because ELU becomes
smooth slowly unlike RELU as it was said before


P_3:

we want to compare two Adam and gradient descent optimizers by using
them in our neural network. In this part, we have used ELU as activation function, because
in previous part we found out the ELU works better than RELU. The learning rate and decay
parameters have been set 10-4 and 10-6 respectively. We already have discussed Adam and gradient descent in descriptive question. We know
that Adam has been published to overcome some lacks in gradient descent, so we expect
that Adam works better than gradient descent. The results show that
Adam has a good performance during learning of the neural network. Adam benefits an
adaptive learning rate and this is why it is better than conventional gradient descent


P_5:

we have implemented the neural network in two different model. We
implemented the network with one convolution using a kernel size of 7*7 in each layer and
implement other network with 2 convolution using kernel size of 3*3 in each layer. We train
two models.

Using larger kernel size can help the neural network to reach the lowest loss faster in
validation step. However, the test accuracy is decreased due to
that. So, we can use a larger kernel size to save time.


P_6:

In this problem, we are going to use dropout in our algorithm. We have dropped out
25% of data. Dropout layers have a very specific function in neural networks. In the last section,
sometimes we are encountered with over-fitting. Where after training, the weights of the
network are so tuned to the training examples they are given that the network doesn’t
perform well when given new examples. The idea of dropout is simplistic in nature. This
layer “drops out” a random set of activations in that layer by setting them to zero. Simple as
that. Now, what are the benefits of such a simple and seemingly unnecessary and
counterintuitive process? Well, in a way, it forces the network to be redundant. By that, it
means the network should be able to provide the right classification or output for a specific

-example even if some of the activations are dropped out. It makes sure that the network is
not getting too “fitted” to the training data and thus helps alleviate the overfitting problem.
An important note is that this layer is only used during training, and not during test time.
We can see the network can act better using dropout. We have better results in test loss
and test accuracy than not using dropout.

