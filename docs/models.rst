************
Models
************

Here is a summary of the networks that were used across the different tasks.

Baseline 
==========

In `biopy.models.Baseline.py` are contained the models as described by :cite:p:`assignment_paper`.
The method presented in the cited paper involves a Variational AE with an additional MLP classifier 
used to discriminate between labels. So in this file there are different implementations for the MLP head 
with different versions; e.g. the `Adversarial_Classifier` class implements the MLP with a Gradient
Reversal Layer; and there is also another version specific for the A549 dataset.

The class of the VAE to which the MLP head is attached to is `FC_VAE`, of which is present also the 
convolutional version for the image data in the CD4 dataset, `NucleiImgVAE`.


Expression Autoencoders
========================

Files in `biopy.models.ExprAutoEncoders.py` includes the different custom models implemented to work with Expression data (mRNA, miRNA, meth...) in 
all the different datasets, and the base layer for all of them is the FC layer. 
We implemented 4 different kinds of AEs:
* Plain AE
* Variational AE
* Adversarial AE
* Supervised Adversarial AE

For each kind the Encoder and Decoder were treated as interchangeable modules.
Infact, the Decoder is the same for all 4 kinds, the `Decoder` class; 
As regards the Encoders, instead, there is the `Encoder` which is made up of FC layers and is suited only for the plain AE.
The `VEncoder` instead is used by the other kinds of AE and returns 2 tensors, representing mean and log variance of the encoded data points.
The `AAE` class implements the Adversarial AE, and it is made up of a `VEncoder`, `Decoder` , and a MLP head
with a gradient reversal layer that acts as discriminator in order to be able to impose a distribution on the latent space.

The Supervised version of the AAE is implemented by `SupervisedAAE`; the way it works is it concatenates 
the one hot encoding of the labels to the latent space before forwarding it to the Discriminator; the purpose
is to be able to impose a distribution conditioned on the label provided.

Both the `AAE` and `SupervisedAAE` class include inside of them the MLP to act as discriminator; however, to be able to use a single
discriminator over multiple AEs (one for each omic), we also implemented a stand-alone discriminator to be instantiated just once and
used to forward data from different AEs. This discriminator come in 2 versions, `ClassDiscriminator` and `ClassDiscriminatorBig`


Image Autoencoders
========================

Files `biopy.models.ImageAutoEncoders.py` and `biopy.models.ImageAAEresnet.py`.
Models dedicated to image processing; for CD4 dataset.
The classes name convention is very similar to the one for Expression Autoencoders.

In `biopy.models.ImageAutoEncoders.py` are implemented 2 kinds of AE:
* Variational AE -> `ImgVAE` class
* Adversarial AE -> `AAEImg` class

They both use the same decoder (`ConvDecoder` class) and encoder that provides means and log variances (`ConvEncoder` class).
Additionally the `AAEImg` model contains the discriminator with gradient reversal.
Even though there is no need of having multiple Image AEs (since there is only one domain for images), for compatibility
reasons, in order to be able to use in the same way all the models, also for images a separate version of the discriminator is provided, not embedded in any model.
This is a simple MLP in the `Discriminator` class.

The `ConvEncoder` is a custom Convolutional networks, with max pooling and batch normalization at every layer; and finally a couple of FC layers
to encode the feature maps to a flat latent space represented by mean and log variance.
For the `ConvDecoder`, in order to 'upscale' from the latent to reconstruct the input image, 2d Transposed Convolutional layer
are used. 

In `biopy.models.ImageAAEresnet.py` we tried a different approach to the implementation of the convolutional AE.
This file contains the implementation of an Adversarial AE, in the class `AAEImgResnet`. 
In this model both the encoder (`ConvEncoder`) and the decoder (`ConvDecoder`) implement the idea of 'skip connection' to help the gradient flow.
To be able to use this mechanism in the Decoder, normal Conv2d layers were alterned with upscaling.


.. warning::
    Some of the classes in the mentioned files overlap with names. (e.g. `ConvEncoder` both in `biopy.models.ImageAutoEncoders.py` and `biopy.models.ImageAAEresnet.py`).
    However these classes are not exported in any outer scope and are only used as building blocks by 
    the full models, which are the ones visible from outside, and their names are properly differentiated. 
