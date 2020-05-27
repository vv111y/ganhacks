- [How to Train a GAN? Tips and tricks to make GANs work](#how-to-train-a-gan-tips-and-tricks-to-make-gans-work)
  - [1: Normalize the inputs](#normalize-the-inputs)
  - [2: A modified loss function](#a-modified-loss-function)
  - [3: Use a spherical Z](#use-a-spherical-z)
  - [4: Batching](#sec-1-4)
  - [5: Normalization](#sec-1-5)
    - [5.a Batch normalization](#a-batch-normalization)
    - [5.b Spectral normalization](#b-layer-normalization)
    - [5.c Instance normalization](#sec-1-5-3)
  - [6: Gradients](#sec-1-6)
    - [6.a: Avoid Sparse Gradients: ReLU, MaxPool](#avoid-sparse-gradients-relu-maxpool)
    - [6.b: Gradient Penalty](#sec-1-6-2)
  - [7: Use Soft and Noisy Labels](#use-soft-and-noisy-labels)
  - [8: DCGAN / Hybrid Models](#dcgan-hybrid-models)
  - [9: Use stability tricks from RL](#use-stability-tricks-from-rl)
  - [10: Use the ADAM Optimizer](#use-the-adam-optimizer)
  - [12: Track failures early](#track-failures-early)
  - [12: Dont balance loss via statistics (unless you have a good reason to)](#dont-balance-loss-via-statistics-unless-you-have-a-good-reason-to)
  - [13: If you have labels, use them](#if-you-have-labels-use-them)
  - [14: Add noise to inputs, decay over time](#add-noise-to-inputs-decay-over-time)
  - [15: Train discriminator more](#train-discriminator-more)
  - [16: [notsure] Batch Discrimination](#notsure-batch-discrimination)
  - [17: Discrete variables in Conditional GANs](#discrete-variables-in-conditional-gans)
  - [18: Use Dropouts in G in both train and test phase](#use-dropouts-in-g-in-both-train-and-test-phase)
  - [19: Sample From G History](#sample-from-g-history)
  - [20: Historical Averaging](#historical-averaging)
  - [Authors](#authors)

As the original *ganhacks* is no longer maintained, this fork is an attempt to keep the information up-to-date with current hacks, at least for 2020.

# How to Train a GAN? Tips and tricks to make GANs work<a id="how-to-train-a-gan-tips-and-tricks-to-make-gans-work"></a>

While research in Generative Adversarial Networks (GANs) continues to improve the fundamental stability of these models, we use a bunch of tricks to train them and make them stable day to day.

Here are a summary of some of the tricks.

[Here's a link to the authors of this document](#authors)

If you find a trick that is particularly useful in practice, please open a Pull Request to add it to the document. If we find it to be reasonable and verified, we will merge it in.

## 1: Normalize the inputs<a id="normalize-the-inputs"></a>

-   normalize the images between -1 and 1
-   Tanh as the last layer of the generator output

## 2: A modified loss function<a id="a-modified-loss-function"></a>

In GAN papers, the loss function to optimize G is `min (log 1-D)`, but in practice folks practically use `max log D` - because the first formulation has vanishing gradients early on - Goodfellow et. al (2014)

In practice, works well: - Flip labels when training generator: real = fake, fake = real

This is called Non-saturating loss, and was confirmed in 2019 paper *A Large-Scale Study on Regularization and Normalization in GANs* (<http://arxiv.org/abs/1807.04720>)

## 3: Use a spherical Z<a id="use-a-spherical-z"></a>

-   Dont sample from a Uniform distribution

![img](images/cube.png "cube")

-   Sample from a gaussian distribution

![img](images/sphere.png "sphere")

-   When doing interpolations, do the interpolation via a great circle, rather than a straight line from point A to point B
-   Tom White's [Sampling Generative Networks](https://arxiv.org/abs/1609.04468) ref code <https://github.com/dribnet/plat> has more details

## 4: Batching<a id="sec-1-4"></a>

-   Construct different mini-batches for real and fake, i.e. each mini-batch needs to contain only all real images or all generated images. Do not mix the different sets together.

![img](images/batchmix.png "batchmix")

## 5: Normalization<a id="sec-1-5"></a>

### 5.a Batch normalization<a id="a-batch-normalization"></a>

-   Apply batch normalization to all but the last layers for both the generator and discriminator

### 5.b Spectral normalization<a id="b-layer-normalization"></a>

-   Highly recommended in *A Large-Scale Study on Regularization and Normalization in GANs* (<http://arxiv.org/abs/1807.04720>)
    -   use spectral normalization on discriminator (<https://arxiv.org/abs/1802.05957>)
    -   this can also be used in the generator (<http://arxiv.org/abs/1805.08318>)
-   Spectral normalization reduces the magnitude of the weight updates:
    -   you will need to train longer, or update learning rate
    -   if SN is only applied to one neural-net then you should either increase its learning rate, or increase the number of iterations for it.

In Pytorch (<https://pytorch.org/docs/master/generated/torch.nn.utils.spectral_norm.html>)

```python
import torch.nn as nn
from torch.nn.utils import spectral_norm

def model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        # ...
        self.m = spectral_norm(nn.Linear(20,40))
```

### 5.c Instance normalization<a id="sec-1-5-3"></a>

-   instance normalization is also an option (for each sample, subtract mean and divide by standard deviation).
    -   Instance normalization can significant improve style based output, see: Instance Normalization: The Missing Ingredient for Fast Stylization (<http://arxiv.org/abs/1607.08022>)
    -   For large models such as CycleGAN it is done out of necessaty (if any normalization is to be done at all). Given GPU RAM constraints only one instance maybe available.

## 6: Gradients<a id="sec-1-6"></a>

### 6.a: Avoid Sparse Gradients: ReLU, MaxPool<a id="avoid-sparse-gradients-relu-maxpool"></a>

-   the stability of the GAN game suffers if you have sparse gradients
-   LeakyReLU = good (in both G and D)
-   For Downsampling, use: Average Pooling, Conv2d + stride
-   For Upsampling, use: PixelShuffle, ConvTranspose2d + stride
    -   PixelShuffle: <https://arxiv.org/abs/1609.05158>

### 6.b: Gradient Penalty<a id="sec-1-6-2"></a>

-   see *Improved Training of Wasserstein GANs* (<https://arxiv.org/abs/1704.00028>)
    -   confirmed in *A Large-Scale Study on Regularization and Normalization in GANs* (<http://arxiv.org/abs/1807.04720>)
-   Pytorch implementation: <https://github.com/EmilienDupont/wgan-gp>
-   TensorFlow implementation: <https://github.com/igul222/improved_wgan_training>
-   Note: this has a higher computational cost

## 7: Use Soft and Noisy Labels<a id="use-soft-and-noisy-labels"></a>

-   Label Smoothing, i.e. if you have two target labels: Real=1 and Fake=0, then for each incoming sample, if it is real, then replace the label with a random number between 0.7 and 1.2, and if it is a fake sample, replace it with 0.0 and 0.3 (for example).
    -   Salimans et. al. 2016

-   make the labels the noisy for the discriminator: occasionally flip the labels when training the discriminator

## 8: DCGAN / Hybrid Models<a id="dcgan-hybrid-models"></a>

-   Use DCGAN when you can. It works!
-   if you cant use DCGANs and no model is stable, use a hybrid model : KL
    -   GAN or VAE + GAN

## 9: Use stability tricks from RL<a id="use-stability-tricks-from-rl"></a>

-   Experience Replay
    -   Keep a replay buffer of past generations and occassionally show them
    -   Keep checkpoints from the past of G and D and occassionaly swap them out for a few iterations

-   All stability tricks that work for deep deterministic policy gradients
-   See Pfau & Vinyals (2016)

## 10: Use the ADAM Optimizer<a id="use-the-adam-optimizer"></a>

-   optim.Adam rules!
    -   See Radford et. al. 2015

-   Use SGD for discriminator and ADAM for generator

## 12: Track failures early<a id="track-failures-early"></a>

-   D loss goes to 0: failure mode
-   check norms of gradients: if they are over 100 things are screwing up
-   when things are working, D loss has low variance and goes down over time vs having huge variance and spiking
-   if loss of generator steadily decreases, then it's fooling D with garbage (says martin)

## 12: Dont balance loss via statistics (unless you have a good reason to)<a id="dont-balance-loss-via-statistics-unless-you-have-a-good-reason-to"></a>

-   Dont try to find a (number of G / number of D) schedule to uncollapse training
-   It's hard and we've all tried it.
-   If you do try it, have a principled approach to it, rather than intuition

For example

    while lossD > A:
      train D
    while lossG > B:
      train G

## 13: If you have labels, use them<a id="if-you-have-labels-use-them"></a>

-   if you have labels available, training the discriminator to also classify the samples: auxillary GANs

## 14: Add noise to inputs, decay over time<a id="add-noise-to-inputs-decay-over-time"></a>

-   Add some artificial noise to inputs to D (Arjovsky et. al., Huszar, 2016)
    -   <http://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/>
    -   <https://openreview.net/forum?id=Hk4_qw5xe>

-   adding gaussian noise to every layer of generator (Zhao et. al. EBGAN)
    -   Improved GANs: OpenAI code also has it (commented out)

## 15: Train discriminator more<a id="train-discriminator-more"></a>

-   especially when you have noise
-   hard to find a schedule of number of D iterations vs G iterations
-   train discriminator in n times using Wasserstein distance
    -   also makes losses correlates with sample quality

## 16: [notsure] Batch Discrimination<a id="notsure-batch-discrimination"></a>

-   Mixed results

## 17: Discrete variables in Conditional GANs<a id="discrete-variables-in-conditional-gans"></a>

-   Use an Embedding layer
-   Add as additional channels to images
-   Keep embedding dimensionality low and upsample to match image channel size

## 18: Use Dropouts in G in both train and test phase<a id="use-dropouts-in-g-in-both-train-and-test-phase"></a>

-   Provide noise in the form of dropout (50%).
-   Apply on several layers of our generator at both training and test time
-   <https://arxiv.org/abs/1611.07004v1>

## 19: Sample From G History<a id="sample-from-g-history"></a>

-   For each batch, sample half of the images from the current generator and half from a history of generated images
-   Section 2.3 <https://arxiv.org/abs/1612.07828>

## 20: Historical Averaging<a id="historical-averaging"></a>

-   Use a historical average of learned parameters (complements #18)
-   Section 3.3 <https://arxiv.org/abs/1606.03498>

## Authors<a id="authors"></a>

-   Soumith Chintala
-   Emily Denton
-   Martin Arjovsky
-   Michael Mathieu
-   Contributors
