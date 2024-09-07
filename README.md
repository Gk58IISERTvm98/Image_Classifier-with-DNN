# All About ANN and DNN

This repository contains implementations and tutorials for training deep neural networks using various architectures, activation functions, optimizers, normalization techniques, dropout, and more. The goal is to provide an in-depth understanding of the essential components required for training state-of-the-art neural networks.

## Artificial Neural Networks 
ANNs are computational models inspired by the human brain's neural networks. They are a core component of machine learning and deep learning, commonly used in tasks such as image classification, speech recognition, natural language processing, and more.

An ANN is made up of layers of interconnected nodes (neurons), where each connection represents a weight that gets adjusted during the learning process. These layers are categorized as input layers, hidden layers, and output layers.

## Deep Neural Networks
A Deep Neural Network (DNN) is simply an ANN with multiple hidden layers. While ANNs can learn basic patterns in the data, DNNs can learn more complex patterns and hierarchical representations because of their depth. The term "deep" refers to the increased number of hidden layers in the network. DNNs are particularly powerful for tasks like image recognition, language translation, and speech recognition.
This repository explores the key components required for training neural networks effectively :

- A variety of **activation functions** for non-linearity.
- Different **optimizers** for gradient descent.
- Techniques for **normalization** and **regularization** to stabilize and improve training.
- Implementations of **dropout** and other regularization strategies.
- Various **loss functions** depending on the task.

## Activation Functions
Activation functions introduce non-linearity to the neural network, allowing it to learn complex patterns. Here are some commonly used activation functions:
-  **Sigmoid :**  $\large\sigma(x) = \Large\frac{1}{1 + e^{-x}}$
-  **tanh :**  $\large\tanh(x) = \Large\frac{e^x - e^{-x}}{e^x + e^{-x}}$
-  **ReLU (Rectified Linear Unit) :** $ReLu(x) = max(0, x)$
-  **Leaky ReLU :** $Leaky ReLU(x) = max(\alpha x, x)$
-  **ELU (Exponential Linear Unit) :**  ELU(x) = x if x>0, and $\alpha (e^x - 1)$ if x<0
-  **Swish:**  $swish (x) = \large x.\sigma (x)$

## Optimizers
Optimizers are crucial for efficiently converging towards the global minimum during training. Common optimizers include:

- Stochastic Gradient Descent (SGD)
- Adam (Adaptive Moment Estimation)
- RMSprop (Root Mean Square Propagation)
- Adagrad
- Adadelta
- Nadam (Nesterov-accelerated Adam)
  
Each optimizer has its strengths depending on the problem at hand. See the optimizers/ directory for detailed implementations and explanations.

## Normalization Techniques
Normalization is essential for stabilizing and speeding up the training of neural networks. The following normalization techniques are commonly used:

- Batch Normalization
- Layer Normalization
- Instance Normalization
- Group Normalization
Normalization helps reduce internal covariate shifts, leading to faster convergence.

## Regularization Techniques
Regularization prevents overfitting by introducing penalties for large weights in the model. Some of the regularization techniques covered include:

- L1 Regularization (Lasso)
- L2 Regularization (Ridge)
- Elastic Net (Combination of L1 and L2)

## Dropout
Dropout is a popular regularization technique where neurons are randomly "dropped" during training to prevent overfitting.

## Loss Functions
Choosing the right loss function is crucial for training neural networks. This repository covers various loss functions, including:
- Mean Squared Error (MSE)
- Cross-Entropy Loss
- Hinge Loss
- Huber Loss
- Kullback-Leibler Divergence
- Different loss functions are suited for different tasks, such as regression, classification, or clustering.
