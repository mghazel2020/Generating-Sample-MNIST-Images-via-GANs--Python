# Generating Sample MNIST Images using Generative Adversarial Networks (GANs)

<img src="images/GANs.png" width="1000">
  
## 1. Objectives


The objective of this project is to demonstrate the application of Generative Adversarial Networks (GANs) to generate sample MNIST images from noise.

## 2.  Generative Adversarial Networks (GANs)

Generative Adversarial Networks, or GANs for short, are an approach to generative modeling using deep learning methods, such as convolutional neural networks.

Generative modeling is an unsupervised learning task in machine learning that involves automatically discovering and learning the regularities or patterns in input data in such a way that the model can be used to generate or output new examples that plausibly could have been drawn from the original dataset.

GANs are a clever way of training a generative model by framing the problem as a supervised learning problem with two sub-models: the generator model that we train to generate new examples, and the discriminator model that tries to classify examples as either real (from the domain) or fake (generated). The two models are trained together in a zero-sum game, adversarial, until the discriminator model is fooled about half the time, meaning the generator model is generating plausible examples.

GANs are an exciting and rapidly changing field, delivering on the promise of generative models in their ability to generate realistic examples across a range of problem domains, most notably in image-to-image translation tasks such as translating photos of summer to winter or day to night, and in generating photorealistic photos of objects, scenes, and people that even humans cannot tell are fake.


In this project, we shall illustrate how to generate sample MNIST images, using Generative Adversarial Networks (GANs).

## 3. Data

We shall illustrate the PCA representation of the  MNIST database of handwritten digits, available from this page, which has a training set of 60,000 examples, and a test set of 10,000 examples. We shall illustrate sample images from this data sets in the next section

## 4. Development

* Project: Generating sample MNIST Hand-written images using Generative Adversarial Networks (GANs):

  * The objective of this project is to demonstrate how to generate sample MNIST hand-written images from noise, using Generative Adversarial Networks (GANS).
  * Author: Mohsen Ghazel (mghazel)
  * Date: April 9th, 2021

### 4.1. Part 1: Python imports and global variables:

#### 4.1.1. Standard scientific Python imports:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># numpy</span>
<span style="color:#200080; font-weight:bold; ">import</span> numpy <span style="color:#200080; font-weight:bold; ">as</span> np
<span style="color:#595979; "># matplotlib</span>
<span style="color:#200080; font-weight:bold; ">import</span> matplotlib<span style="color:#308080; ">.</span>pyplot <span style="color:#200080; font-weight:bold; ">as</span> plt
<span style="color:#595979; "># - import sklearn to use the confusion matrix function</span>
<span style="color:#200080; font-weight:bold; ">from</span> sklearn<span style="color:#308080; ">.</span>metrics <span style="color:#200080; font-weight:bold; ">import</span> confusion_matrix
<span style="color:#595979; "># import imread</span>
<span style="color:#200080; font-weight:bold; ">from</span> skimage<span style="color:#308080; ">.</span>io <span style="color:#200080; font-weight:bold; ">import</span> imread
<span style="color:#595979; "># import itertools</span>
<span style="color:#200080; font-weight:bold; ">import</span> itertools
<span style="color:#595979; "># opencv</span>
<span style="color:#200080; font-weight:bold; ">import</span> cv2
<span style="color:#595979; "># tensorflow</span>
<span style="color:#200080; font-weight:bold; ">import</span> tensorflow <span style="color:#200080; font-weight:bold; ">as</span> tf
<span style="color:#595979; "># pandas imports</span>
<span style="color:#200080; font-weight:bold; ">import</span> pandas <span style="color:#200080; font-weight:bold; ">as</span> pd

<span style="color:#595979; "># keras input layer</span>
<span style="color:#200080; font-weight:bold; ">from</span> tensorflow<span style="color:#308080; ">.</span>keras<span style="color:#308080; ">.</span>layers <span style="color:#200080; font-weight:bold; ">import</span> <span style="color:#400000; ">Input</span>
<span style="color:#595979; "># keras conv2D layer</span>
<span style="color:#200080; font-weight:bold; ">from</span> tensorflow<span style="color:#308080; ">.</span>keras<span style="color:#308080; ">.</span>layers <span style="color:#200080; font-weight:bold; ">import</span> Conv2D
<span style="color:#595979; "># keras MaxPooling2D layer</span>
<span style="color:#200080; font-weight:bold; ">from</span> tensorflow<span style="color:#308080; ">.</span>keras<span style="color:#308080; ">.</span>layers <span style="color:#200080; font-weight:bold; ">import</span> MaxPooling2D
<span style="color:#595979; "># keras Dense layer</span>
<span style="color:#200080; font-weight:bold; ">from</span> tensorflow<span style="color:#308080; ">.</span>keras<span style="color:#308080; ">.</span>layers <span style="color:#200080; font-weight:bold; ">import</span> Dense
<span style="color:#595979; "># keras Flatten layer</span>
<span style="color:#200080; font-weight:bold; ">from</span> tensorflow<span style="color:#308080; ">.</span>keras<span style="color:#308080; ">.</span>layers <span style="color:#200080; font-weight:bold; ">import</span> Flatten
<span style="color:#595979; "># keras Dropout layer</span>
<span style="color:#200080; font-weight:bold; ">from</span> tensorflow<span style="color:#308080; ">.</span>keras<span style="color:#308080; ">.</span>layers <span style="color:#200080; font-weight:bold; ">import</span> Dropout
<span style="color:#595979; "># keras model</span>
<span style="color:#200080; font-weight:bold; ">from</span> tensorflow<span style="color:#308080; ">.</span>keras<span style="color:#308080; ">.</span>models <span style="color:#200080; font-weight:bold; ">import</span> Model
<span style="color:#595979; "># keras sequential model</span>
<span style="color:#200080; font-weight:bold; ">from</span> tensorflow<span style="color:#308080; ">.</span>keras<span style="color:#308080; ">.</span>models <span style="color:#200080; font-weight:bold; ">import</span> Sequential
<span style="color:#595979; "># keras LeakyReLU layer</span>
<span style="color:#200080; font-weight:bold; ">from</span> tensorflow<span style="color:#308080; ">.</span>keras<span style="color:#308080; ">.</span>layers <span style="color:#200080; font-weight:bold; ">import</span> LeakyReLU
<span style="color:#595979; "># keras LeakyReLU layer</span>
<span style="color:#200080; font-weight:bold; ">from</span> tensorflow<span style="color:#308080; ">.</span>keras<span style="color:#308080; ">.</span>layers <span style="color:#200080; font-weight:bold; ">import</span> BatchNormalization
<span style="color:#595979; "># optimizers</span>
<span style="color:#595979; "># SGD</span>
<span style="color:#200080; font-weight:bold; ">from</span> tensorflow<span style="color:#308080; ">.</span>keras<span style="color:#308080; ">.</span>optimizers <span style="color:#200080; font-weight:bold; ">import</span> SGD
<span style="color:#595979; "># Adam</span>
<span style="color:#200080; font-weight:bold; ">from</span> tensorflow<span style="color:#308080; ">.</span>keras<span style="color:#308080; ">.</span>optimizers <span style="color:#200080; font-weight:bold; ">import</span> Adam
<span style="color:#595979; "># random number generators values</span>
<span style="color:#595979; "># seed for reproducing the random number generation</span>
<span style="color:#200080; font-weight:bold; ">from</span> random <span style="color:#200080; font-weight:bold; ">import</span> seed
<span style="color:#595979; "># random integers: I(0,M)</span>
<span style="color:#200080; font-weight:bold; ">from</span> random <span style="color:#200080; font-weight:bold; ">import</span> randint
<span style="color:#595979; "># random standard unform: U(0,1)</span>
<span style="color:#200080; font-weight:bold; ">from</span> random <span style="color:#200080; font-weight:bold; ">import</span> random
<span style="color:#595979; "># time</span>
<span style="color:#200080; font-weight:bold; ">import</span> datetime
<span style="color:#595979; "># I/O</span>
<span style="color:#200080; font-weight:bold; ">import</span> os
<span style="color:#595979; "># sys</span>
<span style="color:#200080; font-weight:bold; ">import</span> sys
</pre>



<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># check for successful package imports and versions</span>
<span style="color:#595979; "># python</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Python version : {0} "</span><span style="color:#308080; ">.</span>format<span style="color:#308080; ">(</span>sys<span style="color:#308080; ">.</span>version<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># OpenCV</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"OpenCV version : {0} "</span><span style="color:#308080; ">.</span>format<span style="color:#308080; ">(</span>cv2<span style="color:#308080; ">.</span>__version__<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># numpy</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Numpy version  : {0}"</span><span style="color:#308080; ">.</span>format<span style="color:#308080; ">(</span>np<span style="color:#308080; ">.</span>__version__<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># tensorflow</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Tensorflow version  : {0}"</span><span style="color:#308080; ">.</span>format<span style="color:#308080; ">(</span>tf<span style="color:#308080; ">.</span>__version__<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>

Python version <span style="color:#308080; ">:</span> <span style="color:#008000; ">3.7</span><span style="color:#308080; ">.</span><span style="color:#008c00; ">10</span> <span style="color:#308080; ">(</span>default<span style="color:#308080; ">,</span> May  <span style="color:#008c00; ">3</span> <span style="color:#008c00; ">2021</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">02</span><span style="color:#308080; ">:</span><span style="color:#008c00; ">48</span><span style="color:#308080; ">:</span><span style="color:#008c00; ">31</span><span style="color:#308080; ">)</span> 
<span style="color:#308080; ">[</span>GCC <span style="color:#008000; ">7.5</span><span style="color:#308080; ">.</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span> 
OpenCV version <span style="color:#308080; ">:</span> <span style="color:#008000; ">4.1</span><span style="color:#308080; ">.</span><span style="color:#008c00; ">2</span> 
Numpy version  <span style="color:#308080; ">:</span> <span style="color:#008000; ">1.19</span><span style="color:#308080; ">.</span><span style="color:#008c00; ">5</span>
Tensorflow version  <span style="color:#308080; ">:</span> <span style="color:#008000; ">2.4</span><span style="color:#308080; ">.</span><span style="color:#008c00; ">1</span>
</pre>


### 4.1.2. Global variables:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># -set the random_state seed = 100 for reproducibilty</span>
random_state_seed <span style="color:#308080; ">=</span> <span style="color:#008c00; ">100</span>

<span style="color:#595979; "># the number of visualized images</span>
num_visualized_images <span style="color:#308080; ">=</span> <span style="color:#008c00; ">25</span>

<span style="color:#595979; ">#------------------------------------------</span>
<span style="color:#595979; "># GANs Hyper-parameters Parameters:</span>
<span style="color:#595979; ">#------------------------------------------</span>
<span style="color:#595979; "># Dimensionality of the latent space</span>
latent_dim <span style="color:#308080; ">=</span> <span style="color:#008c00; ">100</span>
<span style="color:#595979; ">#------------------------------------------</span>
<span style="color:#595979; "># Config parameters</span>
<span style="color:#595979; "># batch-sze</span>
batch_size <span style="color:#308080; ">=</span> <span style="color:#008c00; ">32</span>
<span style="color:#595979; "># the number of epochs</span>
epochs <span style="color:#308080; ">=</span> <span style="color:#008c00; ">30000</span>
<span style="color:#595979; "># after every sample_period steps generate </span>
<span style="color:#595979; "># and save some data</span>
sample_period <span style="color:#308080; ">=</span> <span style="color:#008c00; ">500</span> 
<span style="color:#595979; ">#------------------------------------------</span>
</pre>

### 4.2 Part 2: Load MNIST Dataset:

#### 4.2.1. Load the MNIST dataset :

* Load the MNIST dataset of handwritten digits:
  * 60,000 labelled training examples
  * 10,000 labelled test examples
  * Each handwritten example is 28x28 pixels binary image.


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># Load in the data</span>
mnist <span style="color:#308080; ">=</span> tf<span style="color:#308080; ">.</span>keras<span style="color:#308080; ">.</span>datasets<span style="color:#308080; ">.</span>mnist
<span style="color:#595979; "># load the training and test data</span>
<span style="color:#308080; ">(</span>x_train<span style="color:#308080; ">,</span> y_train<span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> <span style="color:#308080; ">(</span>x_test<span style="color:#308080; ">,</span> y_test<span style="color:#308080; ">)</span> <span style="color:#308080; ">=</span> mnist<span style="color:#308080; ">.</span>load_data<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># map inputs to (-1, +1) for better training</span>
x_train<span style="color:#308080; ">,</span> x_test <span style="color:#308080; ">=</span> x_train <span style="color:#44aadd; ">/</span> <span style="color:#008000; ">255.0</span> <span style="color:#44aadd; ">*</span> <span style="color:#008c00; ">2</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">,</span> x_test <span style="color:#44aadd; ">/</span> <span style="color:#008000; ">255.0</span> <span style="color:#44aadd; ">*</span> <span style="color:#008c00; ">2</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">1</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"x_train.shape:"</span><span style="color:#308080; ">,</span> x_train<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">)</span>

Downloading data <span style="color:#200080; font-weight:bold; ">from</span> https<span style="color:#308080; ">:</span><span style="color:#44aadd; ">//</span>storage<span style="color:#308080; ">.</span>googleapis<span style="color:#308080; ">.</span>com<span style="color:#44aadd; ">/</span>tensorflow<span style="color:#44aadd; ">/</span>tf<span style="color:#44aadd; ">-</span>keras<span style="color:#44aadd; ">-</span>datasets<span style="color:#44aadd; ">/</span>mnist<span style="color:#308080; ">.</span>npz
<span style="color:#008c00; ">11493376</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">11490434</span> <span style="color:#308080; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#308080; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">0</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">0</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">us</span><span style="color:#44aadd; ">/</span>step
x_train<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">:</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">60000</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">28</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">28</span><span style="color:#308080; ">)</span>
</pre>

#### 4.2.2. Examine the shapes of the training and test data:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># Training data:</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># the number of training images</span>
num_train_images <span style="color:#308080; ">=</span> x_train<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"----------------------------------------------"</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Training data:"</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"----------------------------------------------"</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"x_train.shape: "</span><span style="color:#308080; ">,</span> x_train<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Number of training images: "</span><span style="color:#308080; ">,</span> num_train_images<span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Image size: "</span><span style="color:#308080; ">,</span> x_train<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">:</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"----------------------------------------------"</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Classes/labels:"</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"----------------------------------------------"</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'The target labels: '</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>np<span style="color:#308080; ">.</span>unique<span style="color:#308080; ">(</span>y_train<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"----------------------------------------------"</span><span style="color:#308080; ">)</span>

<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># Test data:</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># the number of test images</span>
num_test_images <span style="color:#308080; ">=</span> x_test<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"----------------------------------------------"</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Test data:"</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"----------------------------------------------"</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"x_test.shape: "</span><span style="color:#308080; ">,</span> x_test<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Number of test images: "</span><span style="color:#308080; ">,</span> num_test_images<span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Image size: "</span><span style="color:#308080; ">,</span> x_test<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">:</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"----------------------------------------------"</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Classes/labels:"</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"----------------------------------------------"</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'The target labels: '</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>np<span style="color:#308080; ">.</span>unique<span style="color:#308080; ">(</span>y_train<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"----------------------------------------------"</span><span style="color:#308080; ">)</span>

<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
Training data<span style="color:#308080; ">:</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
x_train<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">:</span>  <span style="color:#308080; ">(</span><span style="color:#008c00; ">60000</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">28</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">28</span><span style="color:#308080; ">)</span>
Number of training images<span style="color:#308080; ">:</span>  <span style="color:#008c00; ">60000</span>
Image size<span style="color:#308080; ">:</span>  <span style="color:#308080; ">(</span><span style="color:#008c00; ">28</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">28</span><span style="color:#308080; ">)</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
Classes<span style="color:#44aadd; ">/</span>labels<span style="color:#308080; ">:</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
The target labels<span style="color:#308080; ">:</span> <span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span> <span style="color:#008c00; ">1</span> <span style="color:#008c00; ">2</span> <span style="color:#008c00; ">3</span> <span style="color:#008c00; ">4</span> <span style="color:#008c00; ">5</span> <span style="color:#008c00; ">6</span> <span style="color:#008c00; ">7</span> <span style="color:#008c00; ">8</span> <span style="color:#008c00; ">9</span><span style="color:#308080; ">]</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
Test data<span style="color:#308080; ">:</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
x_test<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">:</span>  <span style="color:#308080; ">(</span><span style="color:#008c00; ">10000</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">28</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">28</span><span style="color:#308080; ">)</span>
Number of test images<span style="color:#308080; ">:</span>  <span style="color:#008c00; ">10000</span>
Image size<span style="color:#308080; ">:</span>  <span style="color:#308080; ">(</span><span style="color:#008c00; ">28</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">28</span><span style="color:#308080; ">)</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
Classes<span style="color:#44aadd; ">/</span>labels<span style="color:#308080; ">:</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
The target labels<span style="color:#308080; ">:</span> <span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span> <span style="color:#008c00; ">1</span> <span style="color:#008c00; ">2</span> <span style="color:#008c00; ">3</span> <span style="color:#008c00; ">4</span> <span style="color:#008c00; ">5</span> <span style="color:#008c00; ">6</span> <span style="color:#008c00; ">7</span> <span style="color:#008c00; ">8</span> <span style="color:#008c00; ">9</span><span style="color:#308080; ">]</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
</pre>

#### 4.2.3. Examine the number of images for each class of the training and testing subsets:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># create a histogram of the number of images in each class/digit:</span>
<span style="color:#200080; font-weight:bold; ">def</span> plot_bar<span style="color:#308080; ">(</span>y<span style="color:#308080; ">,</span> loc<span style="color:#308080; ">=</span><span style="color:#1060b6; ">'left'</span><span style="color:#308080; ">,</span> relative<span style="color:#308080; ">=</span><span style="color:#074726; ">True</span><span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    width <span style="color:#308080; ">=</span> <span style="color:#008000; ">0.35</span>
    <span style="color:#200080; font-weight:bold; ">if</span> loc <span style="color:#44aadd; ">==</span> <span style="color:#1060b6; ">'left'</span><span style="color:#308080; ">:</span>
        n <span style="color:#308080; ">=</span> <span style="color:#44aadd; ">-</span><span style="color:#008000; ">0.5</span>
    <span style="color:#200080; font-weight:bold; ">elif</span> loc <span style="color:#44aadd; ">==</span> <span style="color:#1060b6; ">'right'</span><span style="color:#308080; ">:</span>
        n <span style="color:#308080; ">=</span> <span style="color:#008000; ">0.5</span>
     
    <span style="color:#595979; "># calculate counts per type and sort, to ensure their order</span>
    unique<span style="color:#308080; ">,</span> counts <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>unique<span style="color:#308080; ">(</span>y<span style="color:#308080; ">,</span> return_counts<span style="color:#308080; ">=</span><span style="color:#074726; ">True</span><span style="color:#308080; ">)</span>
    sorted_index <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>argsort<span style="color:#308080; ">(</span>unique<span style="color:#308080; ">)</span>
    unique <span style="color:#308080; ">=</span> unique<span style="color:#308080; ">[</span>sorted_index<span style="color:#308080; ">]</span>
     
    <span style="color:#200080; font-weight:bold; ">if</span> relative<span style="color:#308080; ">:</span>
        <span style="color:#595979; "># plot as a percentage</span>
        counts <span style="color:#308080; ">=</span> <span style="color:#008c00; ">100</span><span style="color:#44aadd; ">*</span>counts<span style="color:#308080; ">[</span>sorted_index<span style="color:#308080; ">]</span><span style="color:#44aadd; ">/</span><span style="color:#400000; ">len</span><span style="color:#308080; ">(</span>y<span style="color:#308080; ">)</span>
        ylabel_text <span style="color:#308080; ">=</span> <span style="color:#1060b6; ">'% count'</span>
    <span style="color:#200080; font-weight:bold; ">else</span><span style="color:#308080; ">:</span>
        <span style="color:#595979; "># plot counts</span>
        counts <span style="color:#308080; ">=</span> counts<span style="color:#308080; ">[</span>sorted_index<span style="color:#308080; ">]</span>
        ylabel_text <span style="color:#308080; ">=</span> <span style="color:#1060b6; ">'count'</span>
         
    xtemp <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>arange<span style="color:#308080; ">(</span><span style="color:#400000; ">len</span><span style="color:#308080; ">(</span>unique<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>bar<span style="color:#308080; ">(</span>xtemp <span style="color:#44aadd; ">+</span> n<span style="color:#44aadd; ">*</span>width<span style="color:#308080; ">,</span> counts<span style="color:#308080; ">,</span> align<span style="color:#308080; ">=</span><span style="color:#1060b6; ">'center'</span><span style="color:#308080; ">,</span> alpha<span style="color:#308080; ">=</span><span style="color:#008000; ">.7</span><span style="color:#308080; ">,</span> width<span style="color:#308080; ">=</span>width<span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>xticks<span style="color:#308080; ">(</span>xtemp<span style="color:#308080; ">,</span> unique<span style="color:#308080; ">,</span> rotation<span style="color:#308080; ">=</span><span style="color:#008c00; ">45</span><span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>xlabel<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'digit'</span><span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>ylabel<span style="color:#308080; ">(</span>ylabel_text<span style="color:#308080; ">)</span>
 
plt<span style="color:#308080; ">.</span>suptitle<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Frequency of images per digit'</span><span style="color:#308080; ">)</span>
plot_bar<span style="color:#308080; ">(</span>y_train<span style="color:#308080; ">,</span> loc<span style="color:#308080; ">=</span><span style="color:#1060b6; ">'left'</span><span style="color:#308080; ">)</span>
plot_bar<span style="color:#308080; ">(</span>y_test<span style="color:#308080; ">,</span> loc<span style="color:#308080; ">=</span><span style="color:#1060b6; ">'right'</span><span style="color:#308080; ">)</span>
plt<span style="color:#308080; ">.</span>legend<span style="color:#308080; ">(</span><span style="color:#308080; ">[</span>
    <span style="color:#1060b6; ">'train ({0} images)'</span><span style="color:#308080; ">.</span>format<span style="color:#308080; ">(</span><span style="color:#400000; ">len</span><span style="color:#308080; ">(</span>y_train<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> 
    <span style="color:#1060b6; ">'test ({0} images)'</span><span style="color:#308080; ">.</span>format<span style="color:#308080; ">(</span><span style="color:#400000; ">len</span><span style="color:#308080; ">(</span>y_test<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> 
<span style="color:#308080; ">]</span><span style="color:#308080; ">)</span><span style="color:#308080; ">;</span>
</pre>

<img src="images/Histogram-train-test-data.png" width="1000">

#### 4.2.4. Visualize some of the training and test images and their associated targets:

* First implement a visualization functionality to visualize the number of randomly selected images:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">"""</span>
<span style="color:#595979; "># A utility function to visualize multiple images:</span>
<span style="color:#595979; ">"""</span>
<span style="color:#200080; font-weight:bold; ">def</span> visualize_images_and_labels<span style="color:#308080; ">(</span>num_visualized_images <span style="color:#308080; ">=</span> <span style="color:#008c00; ">25</span><span style="color:#308080; ">,</span> dataset_flag <span style="color:#308080; ">=</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
  <span style="color:#595979; ">"""To visualize images.</span>
<span style="color:#595979; "></span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Keyword arguments:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- num_visualized_images -- the number of visualized images (deafult 25)</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- dataset_flag -- 1: training dataset, 2: test dataset</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Return:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- None</span>
<span style="color:#595979; ">&nbsp;&nbsp;"""</span>
  <span style="color:#595979; ">#--------------------------------------------</span>
  <span style="color:#595979; "># the suplot grid shape:</span>
  <span style="color:#595979; ">#--------------------------------------------</span>
  num_rows <span style="color:#308080; ">=</span> <span style="color:#008c00; ">5</span>
  <span style="color:#595979; "># the number of columns</span>
  num_cols <span style="color:#308080; ">=</span> num_visualized_images <span style="color:#44aadd; ">//</span> num_rows
  <span style="color:#595979; "># setup the subplots axes</span>
  fig<span style="color:#308080; ">,</span> axes <span style="color:#308080; ">=</span> plt<span style="color:#308080; ">.</span>subplots<span style="color:#308080; ">(</span>nrows<span style="color:#308080; ">=</span>num_rows<span style="color:#308080; ">,</span> ncols<span style="color:#308080; ">=</span>num_cols<span style="color:#308080; ">,</span> figsize<span style="color:#308080; ">=</span><span style="color:#308080; ">(</span><span style="color:#008c00; ">8</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">10</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
  <span style="color:#595979; "># set a seed random number generator for reproducible results</span>
  seed<span style="color:#308080; ">(</span>random_state_seed<span style="color:#308080; ">)</span>
  <span style="color:#595979; "># iterate over the sub-plots</span>
  <span style="color:#200080; font-weight:bold; ">for</span> row <span style="color:#200080; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#308080; ">(</span>num_rows<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
      <span style="color:#200080; font-weight:bold; ">for</span> col <span style="color:#200080; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#308080; ">(</span>num_cols<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
        <span style="color:#595979; "># get the next figure axis</span>
        ax <span style="color:#308080; ">=</span> axes<span style="color:#308080; ">[</span>row<span style="color:#308080; ">,</span> col<span style="color:#308080; ">]</span><span style="color:#308080; ">;</span>
        <span style="color:#595979; "># turn-off subplot axis</span>
        ax<span style="color:#308080; ">.</span>set_axis_off<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
        <span style="color:#595979; ">#--------------------------------------------</span>
        <span style="color:#595979; "># if the dataset_flag = 1: Training data set</span>
        <span style="color:#595979; ">#--------------------------------------------</span>
        <span style="color:#200080; font-weight:bold; ">if</span> <span style="color:#308080; ">(</span> dataset_flag <span style="color:#44aadd; ">==</span> <span style="color:#008c00; ">1</span> <span style="color:#308080; ">)</span><span style="color:#308080; ">:</span> 
          <span style="color:#595979; "># generate a random image counter</span>
          counter <span style="color:#308080; ">=</span> randint<span style="color:#308080; ">(</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span>num_train_images<span style="color:#308080; ">)</span>
          <span style="color:#595979; "># get the training image</span>
          image <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>squeeze<span style="color:#308080; ">(</span>x_train<span style="color:#308080; ">[</span>counter<span style="color:#308080; ">,</span><span style="color:#308080; ">:</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span>
          <span style="color:#595979; "># get the target associated with the image</span>
          label <span style="color:#308080; ">=</span> y_train<span style="color:#308080; ">[</span>counter<span style="color:#308080; ">]</span>
        <span style="color:#595979; ">#--------------------------------------------</span>
        <span style="color:#595979; "># dataset_flag = 2: Test data set</span>
        <span style="color:#595979; ">#--------------------------------------------</span>
        <span style="color:#200080; font-weight:bold; ">else</span><span style="color:#308080; ">:</span> 
          <span style="color:#595979; "># generate a random image counter</span>
          counter <span style="color:#308080; ">=</span> randint<span style="color:#308080; ">(</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span>num_test_images<span style="color:#308080; ">)</span>
          <span style="color:#595979; "># get the test image</span>
          image <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>squeeze<span style="color:#308080; ">(</span>x_test<span style="color:#308080; ">[</span>counter<span style="color:#308080; ">,</span><span style="color:#308080; ">:</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span>
          <span style="color:#595979; "># get the target associated with the image</span>
          label <span style="color:#308080; ">=</span> y_test<span style="color:#308080; ">[</span>counter<span style="color:#308080; ">]</span>
        <span style="color:#595979; ">#--------------------------------------------</span>
        <span style="color:#595979; "># display the image</span>
        <span style="color:#595979; ">#--------------------------------------------</span>
        ax<span style="color:#308080; ">.</span>imshow<span style="color:#308080; ">(</span>image<span style="color:#308080; ">,</span> cmap<span style="color:#308080; ">=</span>plt<span style="color:#308080; ">.</span>cm<span style="color:#308080; ">.</span>gray_r<span style="color:#308080; ">,</span> interpolation<span style="color:#308080; ">=</span><span style="color:#1060b6; ">'nearest'</span><span style="color:#308080; ">)</span>
        <span style="color:#595979; "># set the title showing the image label</span>
        ax<span style="color:#308080; ">.</span>set_title<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'y ='</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>label<span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> size <span style="color:#308080; ">=</span> <span style="color:#008c00; ">8</span><span style="color:#308080; ">)</span>
</pre>

##### 4.2.4.1. Visualize some of the training images and their associated targets:

<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># call the function to visualize the training images</span>
visualize_images_and_labels<span style="color:#308080; ">(</span>num_visualized_images<span style="color:#308080; ">,</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span>
</pre>

<img src="images/sample-train-images.png" width="1000">

##### 4.2.4.2. Visualize some of the test images and their associated targets:

<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># call the function to visualize the test images</span>
visualize_images_and_labels<span style="color:#308080; ">(</span>num_visualized_images<span style="color:#308080; ">,</span> <span style="color:#008c00; ">2</span><span style="color:#308080; ">)</span>
</pre>

<img src="images/sample-test-images.png"/ width="1000">

#### 4.2.7. Reshape the training and test images:
* The training and test images are 2D grayscale images of size 28x28 pixels
* They need to be flattened to 1D vectors of size: 28x28


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># Flatten the data</span>
N<span style="color:#308080; ">,</span> H<span style="color:#308080; ">,</span> W <span style="color:#308080; ">=</span> x_train<span style="color:#308080; ">.</span>shape
<span style="color:#595979; "># the vector length</span>
D <span style="color:#308080; ">=</span> H <span style="color:#44aadd; ">*</span> W
<span style="color:#595979; "># reshape the training features</span>
x_train <span style="color:#308080; ">=</span> x_train<span style="color:#308080; ">.</span>reshape<span style="color:#308080; ">(</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">,</span> D<span style="color:#308080; ">)</span>
<span style="color:#595979; "># reshape the test features</span>
x_test <span style="color:#308080; ">=</span> x_test<span style="color:#308080; ">.</span>reshape<span style="color:#308080; ">(</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">,</span> D<span style="color:#308080; ">)</span>
</pre>

##### 4.2.7.1. Examine the reshaped training and test images:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># Training data:</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"----------------------------------------------"</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Training data:"</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"----------------------------------------------"</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"x_train.shape: "</span><span style="color:#308080; ">,</span> x_train<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Number of training images: "</span><span style="color:#308080; ">,</span> num_train_images<span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Image size: "</span><span style="color:#308080; ">,</span> x_train<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">:</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"----------------------------------------------"</span><span style="color:#308080; ">)</span>


<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># Test data:</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># the number of test images</span>
num_test_images <span style="color:#308080; ">=</span> x_test<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"----------------------------------------------"</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Test data:"</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"----------------------------------------------"</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"x_test.shape: "</span><span style="color:#308080; ">,</span> x_test<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Number of test images: "</span><span style="color:#308080; ">,</span> num_test_images<span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Image size: "</span><span style="color:#308080; ">,</span> x_test<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">:</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"----------------------------------------------"</span><span style="color:#308080; ">)</span>

<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
Training data<span style="color:#308080; ">:</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
x_train<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">:</span>  <span style="color:#308080; ">(</span><span style="color:#008c00; ">60000</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">784</span><span style="color:#308080; ">)</span>
Number of training images<span style="color:#308080; ">:</span>  <span style="color:#008c00; ">60000</span>
Image size<span style="color:#308080; ">:</span>  <span style="color:#308080; ">(</span><span style="color:#008c00; ">784</span><span style="color:#308080; ">,</span><span style="color:#308080; ">)</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
Test data<span style="color:#308080; ">:</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
x_test<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">:</span>  <span style="color:#308080; ">(</span><span style="color:#008c00; ">10000</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">784</span><span style="color:#308080; ">)</span>
Number of test images<span style="color:#308080; ">:</span>  <span style="color:#008c00; ">10000</span>
Image size<span style="color:#308080; ">:</span>  <span style="color:#308080; ">(</span><span style="color:#008c00; ">784</span><span style="color:#308080; ">,</span><span style="color:#308080; ">)</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
</pre>

### 4.3. Part 3: Construct the GANs Model

#### 4.3.1. The Generator Model:

* Define a function to to design and construct the Generator model:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># Design and construct the Generator model</span>
<span style="color:#200080; font-weight:bold; ">def</span> build_generator<span style="color:#308080; ">(</span>latent_dim<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
  <span style="color:#595979; "># Input layer</span>
  i <span style="color:#308080; ">=</span> <span style="color:#400000; ">Input</span><span style="color:#308080; ">(</span>shape<span style="color:#308080; ">=</span><span style="color:#308080; ">(</span>latent_dim<span style="color:#308080; ">,</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
  <span style="color:#595979; "># Dense layer</span>
  x <span style="color:#308080; ">=</span> Dense<span style="color:#308080; ">(</span><span style="color:#008c00; ">256</span><span style="color:#308080; ">,</span> activation<span style="color:#308080; ">=</span>LeakyReLU<span style="color:#308080; ">(</span>alpha<span style="color:#308080; ">=</span><span style="color:#008000; ">0.2</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span><span style="color:#308080; ">(</span>i<span style="color:#308080; ">)</span>
  <span style="color:#595979; "># BatchNormalization layer</span>
  x <span style="color:#308080; ">=</span> BatchNormalization<span style="color:#308080; ">(</span>momentum<span style="color:#308080; ">=</span><span style="color:#008000; ">0.7</span><span style="color:#308080; ">)</span><span style="color:#308080; ">(</span>x<span style="color:#308080; ">)</span>
  <span style="color:#595979; "># Dense layer</span>
  x <span style="color:#308080; ">=</span> Dense<span style="color:#308080; ">(</span><span style="color:#008c00; ">512</span><span style="color:#308080; ">,</span> activation<span style="color:#308080; ">=</span>LeakyReLU<span style="color:#308080; ">(</span>alpha<span style="color:#308080; ">=</span><span style="color:#008000; ">0.2</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span><span style="color:#308080; ">(</span>x<span style="color:#308080; ">)</span>
  <span style="color:#595979; "># BatchNormalization layer</span>
  x <span style="color:#308080; ">=</span> BatchNormalization<span style="color:#308080; ">(</span>momentum<span style="color:#308080; ">=</span><span style="color:#008000; ">0.7</span><span style="color:#308080; ">)</span><span style="color:#308080; ">(</span>x<span style="color:#308080; ">)</span>
  <span style="color:#595979; "># Dense layer</span>
  x <span style="color:#308080; ">=</span> Dense<span style="color:#308080; ">(</span><span style="color:#008c00; ">1024</span><span style="color:#308080; ">,</span> activation<span style="color:#308080; ">=</span>LeakyReLU<span style="color:#308080; ">(</span>alpha<span style="color:#308080; ">=</span><span style="color:#008000; ">0.2</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span><span style="color:#308080; ">(</span>x<span style="color:#308080; ">)</span>
  <span style="color:#595979; "># BatchNormalization layer</span>
  x <span style="color:#308080; ">=</span> BatchNormalization<span style="color:#308080; ">(</span>momentum<span style="color:#308080; ">=</span><span style="color:#008000; ">0.7</span><span style="color:#308080; ">)</span><span style="color:#308080; ">(</span>x<span style="color:#308080; ">)</span>
  <span style="color:#595979; "># Dense layer</span>
  x <span style="color:#308080; ">=</span> Dense<span style="color:#308080; ">(</span>D<span style="color:#308080; ">,</span> activation<span style="color:#308080; ">=</span><span style="color:#1060b6; ">'tanh'</span><span style="color:#308080; ">)</span><span style="color:#308080; ">(</span>x<span style="color:#308080; ">)</span>
  <span style="color:#595979; "># constuct the Generator model</span>
  model <span style="color:#308080; ">=</span> Model<span style="color:#308080; ">(</span>i<span style="color:#308080; ">,</span> x<span style="color:#308080; ">)</span>
  <span style="color:#595979; "># return the constructed model</span>
  <span style="color:#200080; font-weight:bold; ">return</span> model
</pre>

#### 4.3.2. The Discriminator Model:

* Define a function to design and construct the Discriminator model:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># Design and construct the Discriminator model</span>
<span style="color:#200080; font-weight:bold; ">def</span> build_discriminator<span style="color:#308080; ">(</span>img_size<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
  <span style="color:#595979; "># Input layer</span>
  i <span style="color:#308080; ">=</span> <span style="color:#400000; ">Input</span><span style="color:#308080; ">(</span>shape<span style="color:#308080; ">=</span><span style="color:#308080; ">(</span>img_size<span style="color:#308080; ">,</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
  <span style="color:#595979; "># Dense layer</span>
  x <span style="color:#308080; ">=</span> Dense<span style="color:#308080; ">(</span><span style="color:#008c00; ">512</span><span style="color:#308080; ">,</span> activation<span style="color:#308080; ">=</span>LeakyReLU<span style="color:#308080; ">(</span>alpha<span style="color:#308080; ">=</span><span style="color:#008000; ">0.2</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span><span style="color:#308080; ">(</span>i<span style="color:#308080; ">)</span>
  <span style="color:#595979; "># Dense layer</span>
  x <span style="color:#308080; ">=</span> Dense<span style="color:#308080; ">(</span><span style="color:#008c00; ">256</span><span style="color:#308080; ">,</span> activation<span style="color:#308080; ">=</span>LeakyReLU<span style="color:#308080; ">(</span>alpha<span style="color:#308080; ">=</span><span style="color:#008000; ">0.2</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span><span style="color:#308080; ">(</span>x<span style="color:#308080; ">)</span>
  <span style="color:#595979; "># Dense layer</span>
  x <span style="color:#308080; ">=</span> Dense<span style="color:#308080; ">(</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">,</span> activation<span style="color:#308080; ">=</span><span style="color:#1060b6; ">'sigmoid'</span><span style="color:#308080; ">)</span><span style="color:#308080; ">(</span>x<span style="color:#308080; ">)</span>
  <span style="color:#595979; "># constuct the Generator model</span>
  model <span style="color:#308080; ">=</span> Model<span style="color:#308080; ">(</span>i<span style="color:#308080; ">,</span> x<span style="color:#308080; ">)</span>
  <span style="color:#595979; "># return the constructed model</span>
  <span style="color:#200080; font-weight:bold; ">return</span> model
</pre>

### 4.4 Part 4: Construct and compile the GANs Models

#### 4.4.1. Construct and compile the Discriminator Model:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#-----------------------------------------</span>
<span style="color:#595979; "># Build and compile the discriminator</span>
<span style="color:#595979; ">#-----------------------------------------</span>
<span style="color:#595979; "># Build the Discriminator model</span>
discriminator <span style="color:#308080; ">=</span> build_discriminator<span style="color:#308080; ">(</span>D<span style="color:#308080; ">)</span>

<span style="color:#595979; "># Compile the Discriminator model</span>
discriminator<span style="color:#308080; ">.</span><span style="color:#400000; ">compile</span><span style="color:#308080; ">(</span>
    loss<span style="color:#308080; ">=</span><span style="color:#1060b6; ">'binary_crossentropy'</span><span style="color:#308080; ">,</span>
    optimizer<span style="color:#308080; ">=</span>Adam<span style="color:#308080; ">(</span><span style="color:#008000; ">0.0002</span><span style="color:#308080; ">,</span> <span style="color:#008000; ">0.5</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span>
    metrics<span style="color:#308080; ">=</span><span style="color:#308080; ">[</span><span style="color:#1060b6; ">'accuracy'</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span>
</pre>

### 4.4.2. Construct and compile the Combined Model:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#-----------------------------------------</span>
<span style="color:#595979; "># Build and compile the combined model</span>
<span style="color:#595979; ">#-----------------------------------------</span>
<span style="color:#595979; "># Build the Generator model</span>
generator <span style="color:#308080; ">=</span> build_generator<span style="color:#308080; ">(</span>latent_dim<span style="color:#308080; ">)</span>

<span style="color:#595979; "># Create an input to represent noise sample from latent space</span>
z <span style="color:#308080; ">=</span> <span style="color:#400000; ">Input</span><span style="color:#308080; ">(</span>shape<span style="color:#308080; ">=</span><span style="color:#308080; ">(</span>latent_dim<span style="color:#308080; ">,</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>

<span style="color:#595979; "># Pass noise through generator to get an image</span>
img <span style="color:#308080; ">=</span> generator<span style="color:#308080; ">(</span>z<span style="color:#308080; ">)</span>

<span style="color:#595979; "># Make sure only the generator is trained</span>
discriminator<span style="color:#308080; ">.</span>trainable <span style="color:#308080; ">=</span> <span style="color:#074726; ">False</span>

<span style="color:#595979; "># The true output is fake, but we label them real!</span>
fake_pred <span style="color:#308080; ">=</span> discriminator<span style="color:#308080; ">(</span>img<span style="color:#308080; ">)</span>

<span style="color:#595979; "># Create the combined model object</span>
combined_model <span style="color:#308080; ">=</span> Model<span style="color:#308080; ">(</span>z<span style="color:#308080; ">,</span> fake_pred<span style="color:#308080; ">)</span>

<span style="color:#595979; "># Compile the combined model</span>
combined_model<span style="color:#308080; ">.</span><span style="color:#400000; ">compile</span><span style="color:#308080; ">(</span>loss<span style="color:#308080; ">=</span><span style="color:#1060b6; ">'binary_crossentropy'</span><span style="color:#308080; ">,</span> optimizer<span style="color:#308080; ">=</span>Adam<span style="color:#308080; ">(</span><span style="color:#008000; ">0.0002</span><span style="color:#308080; ">,</span> <span style="color:#008000; ">0.5</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
</pre>

### 4.5. Part 5: Train the GANs Model:

#### 4.5.1 First implement a utility function to generate random samples from the generator:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># A function to generate a grid of 5x5 random samples from the generator</span>
<span style="color:#595979; "># and save them to a file</span>
<span style="color:#200080; font-weight:bold; ">def</span> sample_images<span style="color:#308080; ">(</span>epoch<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
  rows<span style="color:#308080; ">,</span> cols <span style="color:#308080; ">=</span> <span style="color:#008c00; ">5</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">5</span>
  noise <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>random<span style="color:#308080; ">.</span>randn<span style="color:#308080; ">(</span>rows <span style="color:#44aadd; ">*</span> cols<span style="color:#308080; ">,</span> latent_dim<span style="color:#308080; ">)</span>
  imgs <span style="color:#308080; ">=</span> generator<span style="color:#308080; ">.</span>predict<span style="color:#308080; ">(</span>noise<span style="color:#308080; ">)</span>

  <span style="color:#595979; "># Rescale images 0 - 1</span>
  imgs <span style="color:#308080; ">=</span> <span style="color:#008000; ">0.5</span> <span style="color:#44aadd; ">*</span> imgs <span style="color:#44aadd; ">+</span> <span style="color:#008000; ">0.5</span>

  fig<span style="color:#308080; ">,</span> axs <span style="color:#308080; ">=</span> plt<span style="color:#308080; ">.</span>subplots<span style="color:#308080; ">(</span>rows<span style="color:#308080; ">,</span> cols<span style="color:#308080; ">)</span>
  idx <span style="color:#308080; ">=</span> <span style="color:#008c00; ">0</span>
  <span style="color:#200080; font-weight:bold; ">for</span> i <span style="color:#200080; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#308080; ">(</span>rows<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#200080; font-weight:bold; ">for</span> j <span style="color:#200080; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#308080; ">(</span>cols<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
      axs<span style="color:#308080; ">[</span>i<span style="color:#308080; ">,</span>j<span style="color:#308080; ">]</span><span style="color:#308080; ">.</span>imshow<span style="color:#308080; ">(</span>imgs<span style="color:#308080; ">[</span>idx<span style="color:#308080; ">]</span><span style="color:#308080; ">.</span>reshape<span style="color:#308080; ">(</span>H<span style="color:#308080; ">,</span> W<span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> cmap<span style="color:#308080; ">=</span><span style="color:#1060b6; ">'gray'</span><span style="color:#308080; ">)</span>
      axs<span style="color:#308080; ">[</span>i<span style="color:#308080; ">,</span>j<span style="color:#308080; ">]</span><span style="color:#308080; ">.</span>axis<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'off'</span><span style="color:#308080; ">)</span>
      idx <span style="color:#44aadd; ">+</span><span style="color:#308080; ">=</span> <span style="color:#008c00; ">1</span>
  fig<span style="color:#308080; ">.</span>savefig<span style="color:#308080; ">(</span><span style="color:#1060b6; ">"gan_images/%d.png"</span> <span style="color:#44aadd; ">%</span> epoch<span style="color:#308080; ">)</span>
  plt<span style="color:#308080; ">.</span>close<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
</pre>

### 4.5.2. Start training the GANs models:

<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># Create batch labels to use when calling train_on_batch</span>
ones <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>ones<span style="color:#308080; ">(</span>batch_size<span style="color:#308080; ">)</span>
zeros <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>zeros<span style="color:#308080; ">(</span>batch_size<span style="color:#308080; ">)</span>

<span style="color:#595979; "># Store the losses</span>
d_losses <span style="color:#308080; ">=</span> <span style="color:#308080; ">[</span><span style="color:#308080; ">]</span>
g_losses <span style="color:#308080; ">=</span> <span style="color:#308080; ">[</span><span style="color:#308080; ">]</span>

<span style="color:#595979; "># Create a folder to store generated images</span>
<span style="color:#200080; font-weight:bold; ">if</span> <span style="color:#200080; font-weight:bold; ">not</span> os<span style="color:#308080; ">.</span>path<span style="color:#308080; ">.</span>exists<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'gan_images'</span><span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
  os<span style="color:#308080; ">.</span>makedirs<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'gan_images'</span><span style="color:#308080; ">)</span>

<span style="color:#595979; "># Main training loop</span>
<span style="color:#200080; font-weight:bold; ">for</span> epoch <span style="color:#200080; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#308080; ">(</span>epochs<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
  <span style="color:#595979; ">#-------------------------------------------</span>
  <span style="color:#595979; "># Step 1: Train the discriminator model:</span>
  <span style="color:#595979; ">#-------------------------------------------</span>
  <span style="color:#595979; "># Select a random batch of images</span>
  idx <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>random<span style="color:#308080; ">.</span>randint<span style="color:#308080; ">(</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span> x_train<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span><span style="color:#308080; ">,</span> batch_size<span style="color:#308080; ">)</span>
  real_imgs <span style="color:#308080; ">=</span> x_train<span style="color:#308080; ">[</span>idx<span style="color:#308080; ">]</span>
  
  <span style="color:#595979; "># Generate fake images</span>
  noise <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>random<span style="color:#308080; ">.</span>randn<span style="color:#308080; ">(</span>batch_size<span style="color:#308080; ">,</span> latent_dim<span style="color:#308080; ">)</span>
  fake_imgs <span style="color:#308080; ">=</span> generator<span style="color:#308080; ">.</span>predict<span style="color:#308080; ">(</span>noise<span style="color:#308080; ">)</span>
  
  <span style="color:#595979; "># Train the discriminator</span>
  <span style="color:#595979; "># both loss and accuracy are returned</span>
  d_loss_real<span style="color:#308080; ">,</span> d_acc_real <span style="color:#308080; ">=</span> discriminator<span style="color:#308080; ">.</span>train_on_batch<span style="color:#308080; ">(</span>real_imgs<span style="color:#308080; ">,</span> ones<span style="color:#308080; ">)</span>
  d_loss_fake<span style="color:#308080; ">,</span> d_acc_fake <span style="color:#308080; ">=</span> discriminator<span style="color:#308080; ">.</span>train_on_batch<span style="color:#308080; ">(</span>fake_imgs<span style="color:#308080; ">,</span> zeros<span style="color:#308080; ">)</span>
  d_loss <span style="color:#308080; ">=</span> <span style="color:#008000; ">0.5</span> <span style="color:#44aadd; ">*</span> <span style="color:#308080; ">(</span>d_loss_real <span style="color:#44aadd; ">+</span> d_loss_fake<span style="color:#308080; ">)</span>
  d_acc  <span style="color:#308080; ">=</span> <span style="color:#008000; ">0.5</span> <span style="color:#44aadd; ">*</span> <span style="color:#308080; ">(</span>d_acc_real <span style="color:#44aadd; ">+</span> d_acc_fake<span style="color:#308080; ">)</span>
  
  <span style="color:#595979; ">#-------------------------------------------</span>
  <span style="color:#595979; "># Step 2: Train the combined model:</span>
  <span style="color:#595979; ">#-------------------------------------------</span>
  <span style="color:#595979; "># generate random Gaussian normal</span>
  noise <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>random<span style="color:#308080; ">.</span>randn<span style="color:#308080; ">(</span>batch_size<span style="color:#308080; ">,</span> latent_dim<span style="color:#308080; ">)</span>
  <span style="color:#595979; "># train the combined model</span>
  g_loss <span style="color:#308080; ">=</span> combined_model<span style="color:#308080; ">.</span>train_on_batch<span style="color:#308080; ">(</span>noise<span style="color:#308080; ">,</span> ones<span style="color:#308080; ">)</span>
  
  <span style="color:#595979; "># Do it again: generate random Gaussian normal</span>
  noise <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>random<span style="color:#308080; ">.</span>randn<span style="color:#308080; ">(</span>batch_size<span style="color:#308080; ">,</span> latent_dim<span style="color:#308080; ">)</span>
  <span style="color:#595979; "># train the combined model</span>
  g_loss <span style="color:#308080; ">=</span> combined_model<span style="color:#308080; ">.</span>train_on_batch<span style="color:#308080; ">(</span>noise<span style="color:#308080; ">,</span> ones<span style="color:#308080; ">)</span>
  
  <span style="color:#595979; "># Save the losses</span>
  d_losses<span style="color:#308080; ">.</span>append<span style="color:#308080; ">(</span>d_loss<span style="color:#308080; ">)</span>
  g_losses<span style="color:#308080; ">.</span>append<span style="color:#308080; ">(</span>g_loss<span style="color:#308080; ">)</span>
  <span style="color:#595979; "># log the losses for certain epochs</span>
  <span style="color:#200080; font-weight:bold; ">if</span> epoch <span style="color:#44aadd; ">%</span> <span style="color:#008c00; ">100</span> <span style="color:#44aadd; ">==</span> <span style="color:#008c00; ">0</span><span style="color:#308080; ">:</span>
    <span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span>f<span style="color:#1060b6; ">"epoch: {epoch+1}/{epochs}, d_loss: {d_loss:.2f}, </span><span style="color:#0f69ff; ">\</span><span style="color:#1060b6; "></span>
<span style="color:#1060b6; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;d_acc: {d_acc:.2f}, g_loss: {g_loss:.2f}"</span><span style="color:#308080; ">)</span>
  <span style="color:#595979; "># save the images for certain epochs</span>
  <span style="color:#200080; font-weight:bold; ">if</span> epoch <span style="color:#44aadd; ">%</span> sample_period <span style="color:#44aadd; ">==</span> <span style="color:#008c00; ">0</span><span style="color:#308080; ">:</span>
    sample_images<span style="color:#308080; ">(</span>epoch<span style="color:#308080; ">)</span>
</pre>


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;">epoch<span style="color:#308080; ">:</span> <span style="color:#008c00; ">1</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">30000</span><span style="color:#308080; ">,</span> d_loss<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.91</span><span style="color:#308080; ">,</span>       d_acc<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.14</span><span style="color:#308080; ">,</span> g_loss<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.64</span>
epoch<span style="color:#308080; ">:</span> <span style="color:#008c00; ">101</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">30000</span><span style="color:#308080; ">,</span> d_loss<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.16</span><span style="color:#308080; ">,</span>       d_acc<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.97</span><span style="color:#308080; ">,</span> g_loss<span style="color:#308080; ">:</span> <span style="color:#008000; ">2.78</span>
epoch<span style="color:#308080; ">:</span> <span style="color:#008c00; ">201</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">30000</span><span style="color:#308080; ">,</span> d_loss<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.65</span><span style="color:#308080; ">,</span>       d_acc<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.67</span><span style="color:#308080; ">,</span> g_loss<span style="color:#308080; ">:</span> <span style="color:#008000; ">1.26</span>
epoch<span style="color:#308080; ">:</span> <span style="color:#008c00; ">301</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">30000</span><span style="color:#308080; ">,</span> d_loss<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.73</span><span style="color:#308080; ">,</span>       d_acc<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.42</span><span style="color:#308080; ">,</span> g_loss<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.61</span>
epoch<span style="color:#308080; ">:</span> <span style="color:#008c00; ">401</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">30000</span><span style="color:#308080; ">,</span> d_loss<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.70</span><span style="color:#308080; ">,</span>       d_acc<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.47</span><span style="color:#308080; ">,</span> g_loss<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.61</span>
epoch<span style="color:#308080; ">:</span> <span style="color:#008c00; ">501</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">30000</span><span style="color:#308080; ">,</span> d_loss<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.69</span><span style="color:#308080; ">,</span>       d_acc<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.48</span><span style="color:#308080; ">,</span> g_loss<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.63</span>
<span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span>
<span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span>
<span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span>
epoch<span style="color:#308080; ">:</span> <span style="color:#008c00; ">29501</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">30000</span><span style="color:#308080; ">,</span> d_loss<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.65</span><span style="color:#308080; ">,</span>       d_acc<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.64</span><span style="color:#308080; ">,</span> g_loss<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.83</span>
epoch<span style="color:#308080; ">:</span> <span style="color:#008c00; ">29601</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">30000</span><span style="color:#308080; ">,</span> d_loss<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.74</span><span style="color:#308080; ">,</span>       d_acc<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.48</span><span style="color:#308080; ">,</span> g_loss<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.91</span>
epoch<span style="color:#308080; ">:</span> <span style="color:#008c00; ">29701</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">30000</span><span style="color:#308080; ">,</span> d_loss<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.63</span><span style="color:#308080; ">,</span>       d_acc<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.67</span><span style="color:#308080; ">,</span> g_loss<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.94</span>
epoch<span style="color:#308080; ">:</span> <span style="color:#008c00; ">29801</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">30000</span><span style="color:#308080; ">,</span> d_loss<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.68</span><span style="color:#308080; ">,</span>       d_acc<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.59</span><span style="color:#308080; ">,</span> g_loss<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.86</span>
epoch<span style="color:#308080; ">:</span> <span style="color:#008c00; ">29901</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">30000</span><span style="color:#308080; ">,</span> d_loss<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.64</span><span style="color:#308080; ">,</span>       d_acc<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.62</span><span style="color:#308080; ">,</span> g_loss<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.87</span>
</pre>

### 4.6. Part 6: Evaluate the trained Discriminator and Generator models:

#### 4.6.1. Display the Training losses:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># create a figure and set its axis</span>
fig_size <span style="color:#308080; ">=</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">10</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">8</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># create the figure </span>
plt<span style="color:#308080; ">.</span>figure<span style="color:#308080; ">(</span>figsize<span style="color:#308080; ">=</span>fig_size<span style="color:#308080; ">)</span>
<span style="color:#595979; "># plot the generator model losses</span>
plt<span style="color:#308080; ">.</span>plot<span style="color:#308080; ">(</span>g_losses<span style="color:#308080; ">,</span> label<span style="color:#308080; ">=</span><span style="color:#1060b6; ">'Generator-losses'</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># plot the discriminator model losses</span>
plt<span style="color:#308080; ">.</span>plot<span style="color:#308080; ">(</span>d_losses<span style="color:#308080; ">,</span> label<span style="color:#308080; ">=</span><span style="color:#1060b6; ">'Discriminator-losses'</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># add legend</span>
plt<span style="color:#308080; ">.</span>legend<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># add xlabel</span>
plt<span style="color:#308080; ">.</span>xlabel<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Epoch'</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># add ylabel</span>
plt<span style="color:#308080; ">.</span>ylabel<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Loss'</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># add title</span>
plt<span style="color:#308080; ">.</span>title<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Discriminator and Generator Losses vs. Epochs'</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># display the figure</span>
plt<span style="color:#308080; ">.</span>show<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
</pre>

<img src="images/training-losses.png" width="1000">

### 4.7. Part 7: Visualize some of the generated results:

#### 4.7.1. List the saved generated images at different epochs:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># list the content of the saved images folder</span>
!ls gan_images

<span style="color:#008000; ">0.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>	   <span style="color:#008000; ">13500.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>  <span style="color:#008000; ">17500.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>  <span style="color:#008000; ">21500.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>  <span style="color:#008000; ">25500.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>  <span style="color:#008000; ">3000.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>  <span style="color:#008000; ">7000.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>
<span style="color:#008000; ">10000.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>  <span style="color:#008000; ">14000.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>  <span style="color:#008000; ">18000.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>  <span style="color:#008000; ">22000.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>  <span style="color:#008000; ">26000.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>  <span style="color:#008000; ">3500.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>  <span style="color:#008000; ">7500.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>
<span style="color:#008000; ">1000.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>   <span style="color:#008000; ">14500.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>  <span style="color:#008000; ">18500.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>  <span style="color:#008000; ">22500.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>  <span style="color:#008000; ">26500.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>  <span style="color:#008000; ">4000.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>  <span style="color:#008000; ">8000.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>
<span style="color:#008000; ">10500.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>  <span style="color:#008000; ">15000.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>  <span style="color:#008000; ">19000.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>  <span style="color:#008000; ">23000.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>  <span style="color:#008000; ">27000.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>  <span style="color:#008000; ">4500.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>  <span style="color:#008000; ">8500.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>
<span style="color:#008000; ">11000.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>  <span style="color:#008000; ">1500.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>   <span style="color:#008000; ">19500.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>  <span style="color:#008000; ">23500.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>  <span style="color:#008000; ">27500.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>  <span style="color:#008000; ">5000.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>  <span style="color:#008000; ">9000.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>
<span style="color:#008000; ">11500.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>  <span style="color:#008000; ">15500.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>  <span style="color:#008000; ">20000.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>  <span style="color:#008000; ">24000.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>  <span style="color:#008000; ">28000.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>  <span style="color:#008000; ">500.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>	 <span style="color:#008000; ">9500.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>
<span style="color:#008000; ">12000.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>  <span style="color:#008000; ">16000.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>  <span style="color:#008000; ">2000.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>	 <span style="color:#008000; ">24500.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>  <span style="color:#008000; ">28500.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>  <span style="color:#008000; ">5500.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>
<span style="color:#008000; ">12500.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>  <span style="color:#008000; ">16500.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>  <span style="color:#008000; ">20500.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>  <span style="color:#008000; ">25000.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>  <span style="color:#008000; ">29000.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>  <span style="color:#008000; ">6000.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>
<span style="color:#008000; ">13000.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>  <span style="color:#008000; ">17000.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>  <span style="color:#008000; ">21000.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>  <span style="color:#008000; ">2500.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>   <span style="color:#008000; ">29500.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>  <span style="color:#008000; ">6500.</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">png</span>
</pre>

#### 4.7.2.Visualize sample generated images after different epochs:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#------------------------------------------------</span>
<span style="color:#595979; "># Display sample generated images at epoch: 0</span>
<span style="color:#595979; ">#------------------------------------------------</span>
<span style="color:#595979; "># these should be pure Gaussian noise</span>
im <span style="color:#308080; ">=</span> imread<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'gan_images/0.png'</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># create a figure and set its axis</span>
fig_size <span style="color:#308080; ">=</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">16</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">16</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># create the figure </span>
plt<span style="color:#308080; ">.</span>figure<span style="color:#308080; ">(</span>figsize<span style="color:#308080; ">=</span>fig_size<span style="color:#308080; ">)</span>
<span style="color:#595979; "># display the image</span>
plt<span style="color:#308080; ">.</span>imshow<span style="color:#308080; ">(</span>im<span style="color:#308080; ">)</span>
<span style="color:#595979; "># hide the axes</span>
plt<span style="color:#308080; ">.</span>axis<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'off'</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># display the figure</span>
plt<span style="color:#308080; ">.</span>show<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
</pre>

<figure>
<img src="images/iterations-00000.png" width="1000">
<blockquote align="center"> After 0 Iterations </blockquote>
</figure>


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#------------------------------------------------</span>
<span style="color:#595979; "># Display sample generated images at epoch: 1000</span>
<span style="color:#595979; ">#------------------------------------------------</span>
<span style="color:#595979; "># these should resemble hand-written digits</span>
im <span style="color:#308080; ">=</span> imread<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'gan_images/1000.png'</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># create a figure and set its axis</span>
fig_size <span style="color:#308080; ">=</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">16</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">16</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># create the figure </span>
plt<span style="color:#308080; ">.</span>figure<span style="color:#308080; ">(</span>figsize<span style="color:#308080; ">=</span>fig_size<span style="color:#308080; ">)</span>
<span style="color:#595979; "># display the image</span>
plt<span style="color:#308080; ">.</span>imshow<span style="color:#308080; ">(</span>im<span style="color:#308080; ">)</span>
<span style="color:#595979; "># hide the axes</span>
plt<span style="color:#308080; ">.</span>axis<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'off'</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># display the figure</span>
plt<span style="color:#308080; ">.</span>show<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
</pre>

<figure>
<img src="images/iterations-01000.png" width="1000">
<blockquote align="center"> After 1000 Iterations </blockquote>
</figure>


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#------------------------------------------------</span>
<span style="color:#595979; "># Display sample generated images at epoch: 5000</span>
<span style="color:#595979; ">#------------------------------------------------</span>
<span style="color:#595979; "># these should resemble hand-written digits</span>
im <span style="color:#308080; ">=</span> imread<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'gan_images/5000.png'</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># create a figure and set its axis</span>
fig_size <span style="color:#308080; ">=</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">16</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">16</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># create the figure </span>
plt<span style="color:#308080; ">.</span>figure<span style="color:#308080; ">(</span>figsize<span style="color:#308080; ">=</span>fig_size<span style="color:#308080; ">)</span>
<span style="color:#595979; "># display the image</span>
plt<span style="color:#308080; ">.</span>imshow<span style="color:#308080; ">(</span>im<span style="color:#308080; ">)</span>
<span style="color:#595979; "># hide the axes</span>
plt<span style="color:#308080; ">.</span>axis<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'off'</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># display the figure</span>
plt<span style="color:#308080; ">.</span>show<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
</pre>

<figure>
<img src="images/iterations-05000.png" width="1000">
<blockquote align="center"> After 5000 Iterations </blockquote>
</figure>


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#------------------------------------------------</span>
<span style="color:#595979; "># Display sample generated images at epoch: 10000</span>
<span style="color:#595979; ">#------------------------------------------------</span>
<span style="color:#595979; "># these should resemble hand-written digits</span>
im <span style="color:#308080; ">=</span> imread<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'gan_images/10000.png'</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># create a figure and set its axis</span>
fig_size <span style="color:#308080; ">=</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">16</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">16</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># create the figure </span>
plt<span style="color:#308080; ">.</span>figure<span style="color:#308080; ">(</span>figsize<span style="color:#308080; ">=</span>fig_size<span style="color:#308080; ">)</span>
<span style="color:#595979; "># display the image</span>
plt<span style="color:#308080; ">.</span>imshow<span style="color:#308080; ">(</span>im<span style="color:#308080; ">)</span>
<span style="color:#595979; "># hide the axes</span>
plt<span style="color:#308080; ">.</span>axis<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'off'</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># display the figure</span>
plt<span style="color:#308080; ">.</span>show<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
</pre>

<figure>
<img src="images/iterations-10000.png" width="1000">
<blockquote align="center"> After 10000 Iterations </blockquote>
</figure>



<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#------------------------------------------------</span>
<span style="color:#595979; "># Display sample generated images at epoch: 20000</span>
<span style="color:#595979; ">#------------------------------------------------</span>
<span style="color:#595979; "># these should resemble hand-written digits</span>
im <span style="color:#308080; ">=</span> imread<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'gan_images/20000.png'</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># create a figure and set its axis</span>
fig_size <span style="color:#308080; ">=</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">16</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">16</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># create the figure </span>
plt<span style="color:#308080; ">.</span>figure<span style="color:#308080; ">(</span>figsize<span style="color:#308080; ">=</span>fig_size<span style="color:#308080; ">)</span>
<span style="color:#595979; "># display the image</span>
plt<span style="color:#308080; ">.</span>imshow<span style="color:#308080; ">(</span>im<span style="color:#308080; ">)</span>
<span style="color:#595979; "># hide the axes</span>
plt<span style="color:#308080; ">.</span>axis<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'off'</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># display the figure</span>
plt<span style="color:#308080; ">.</span>show<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
</pre>

<figure>
<img src="images/iterations-20000.png" width="1000">
<blockquote align="center"> After 20000 Iterations </blockquote>
</figure>


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#------------------------------------------------</span>
<span style="color:#595979; "># Display sample generated images the end of </span>
<span style="color:#595979; "># training at epoch: 30,0000</span>
<span style="color:#595979; ">#------------------------------------------------</span>
<span style="color:#595979; "># these should resemble hand-written digits</span>
im <span style="color:#308080; ">=</span> imread<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'gan_images/29500.png'</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># create a figure and set its axis</span>
fig_size <span style="color:#308080; ">=</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">16</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">16</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># create the figure </span>
plt<span style="color:#308080; ">.</span>figure<span style="color:#308080; ">(</span>figsize<span style="color:#308080; ">=</span>fig_size<span style="color:#308080; ">)</span>
<span style="color:#595979; "># display the image</span>
plt<span style="color:#308080; ">.</span>imshow<span style="color:#308080; ">(</span>im<span style="color:#308080; ">)</span>
<span style="color:#595979; "># hide the axes</span>
plt<span style="color:#308080; ">.</span>axis<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'off'</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># display the figure</span>
plt<span style="color:#308080; ">.</span>show<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
</pre>


<figure>
<img src="images/iterations-30000.png" width="1000">
<blockquote align="center"> After 30000 Iterations </blockquote>
</figure>

### 3.8. Part 8: Display a successful execution message:



<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># display a final message</span>
<span style="color:#595979; "># current time</span>
now <span style="color:#308080; ">=</span> datetime<span style="color:#308080; ">.</span>datetime<span style="color:#308080; ">.</span>now<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># display a message</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Program executed successfully on: '</span><span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>now<span style="color:#308080; ">.</span>strftime<span style="color:#308080; ">(</span><span style="color:#1060b6; ">"%Y-%m-%d %H:%M:%S"</span><span style="color:#308080; ">)</span> <span style="color:#44aadd; ">+</span> <span style="color:#1060b6; ">"...Goodbye!</span><span style="color:#0f69ff; ">\n</span><span style="color:#1060b6; ">"</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>

Program executed successfully on<span style="color:#308080; ">:</span> <span style="color:#008c00; ">2021</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">05</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">10</span> <span style="color:#008c00; ">03</span><span style="color:#308080; ">:</span><span style="color:#008c00; ">29</span><span style="color:#308080; ">:</span><span style="color:#008000; ">48.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span>Goodbye!
</pre>


## 5. Analysis

* In view of the presented results, we make the following observations:

  * The generated MNIST fake images look very similar to actual MNIST images
  * These images are generated starting from random Gaussian noise from the latent space
  * This quick example demonstrates how GANs perform

## 6. Future Work

* We plan to explore the following related issues:

  * In this example, we implemented GANs as feedforward model
  * We plan to implemented GANs using convolutional layers

## 7. References

1. Yann LeCun et. al. THE MNIST DATABASE of handwritten digits. http://yann.lecun.com/exdb/mnist/ 
2. Jason Brownlee. A Gentle Introduction to Generative Adversarial Networks (GANs). https://machinelearningmastery.com/what-are-generative-adversarial-networks-gans/ 
3. Daniele Paliotta. Introduction to GANs with Python and TensorFlow. https://stackabuse.com/introduction-to-gans-with-python-and-tensorflow/ 
4. Renato Candido. Generative Adversarial Networks: Build Your First Models. https://realpython.com/generative-adversarial-networks/ 
5. Jason Brownlee. Generative Adversarial Networks with Python: Deep Learning Generative Models for Image Synthesis and Image Translation. https://machinelearningmastery.com/generative_adversarial_networks/ 
6. Jason Brownlee. How to Develop a GAN for Generating MNIST Handwritten Digits. https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-an-mnist-handwritten-digits-from-scratch-in-keras/ 
7. Paolo Caressa. How to build a GAN in Python. https://www.codemotion.com/magazine/dev-hub/machine-learning-dev/how-to-build-a-gan-in-python/ 
8. Sadrach Pierre. Generative Adversarial Networks in Python Introduction to GANs in Python. https://towardsdatascience.com/generative-adversarial-networks-in-python-73d3972823d3 
9. Tensorflow. Deep Convolutional Generative Adversarial Network. https://www.tensorflow.org/tutorials/generative/dcgan 
10. Adrian Rosebrock. GANs with Keras and TensorFlow. https://www.pyimagesearch.com/2020/11/16/gans-with-keras-and-tensorflow/ 
11. Diego Gomez Mosquera. GANs from Scratch 1: A deep introduction. With code in PyTorch and TensorFlow. https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f
 














