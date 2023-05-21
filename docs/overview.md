# Frankenstein Network (FranNet)

FranNet is a neural network architecture that combines a CNN encoder with a NeRF-like decoder to perform upscaling, denoising, and colorization of input images. The purpose of this project is to explore the capabilities and limitations of this approach and implement it as a proof of concept. The network consists of an encoder, a decoder, and three restorators: single-step, autoregressive, and diffusion.

FranNet offers several advantages:

1. Arbitrary image size: One key advantage of FranNet is its ability to generate images of any size. This flexibility allows for the production of high-resolution images, enabling various applications that require detailed visual output.

2. Controlled performance during training: FranNet allows for fine-tuning the trade-off between performance and training speed or memory usage. By training the network using only a small fraction of target values, it becomes possible to achieve desired performance levels while optimizing computational resources.

3. Progressive upscaling: FranNet has the capability to implement progressive upscaling, which can enhance the efficiency of the inference process. Although the current version of the network does not include this feature, utilizing diffusion/autoregressive restorators with progressive upscaling could yield notable improvements in terms of speed.

## Main Parts of the Network

<img src="img/scheme.png" width="70%" style="display: block; margin: auto;" />

### Encoder

The encoder is a convolutional neural network (CNN) that takes a downsampled and grayscale input image. It extracts high-level features from the image and generates a compressed representation in the form of a main latent vector and multiple feature maps with different resolutions. These extracted features are then passed to the decoder.

### Decoder

The decoder takes as input the coordinates of a point (x and y in the range 0 to 1) to reconstruct, the main latent vector, and corresponding vectors extracted from the feature maps.

### Restorators

The restorators combine the encoder, decoder, and a set of points to produce the final output. There are three types of restorators: single-step, autoregressive, and diffusion.

## Single step

The single-step restoration process uses the decoder to generate the final output in a single step. This approach is commonly employed in image restoration, but typically, the entire image is available for processing, unlike the FranNet's single-step approach that operates on a single point at a time.

## Autoregressive

Autoregressive restoration is a versatile approach that encompasses both single-step and diffusion methods. It involves using a decoder to gradually restore pixel values in multiple steps, leading to the final output.

One intriguing aspect of the autoregressive approach is its compatibility with the concept of interpolants. In this context, the restoration process can be seen as an interpolation between the correct pixel value (`x0`) and an arbitrary value (`x1`), which depends on a parameter `t`. The interpolation method can be chosen arbitrarily as long as it can produce `x0`.

By disregarding the `t` and `x1` values and solely focusing on producing `x0`, we arrive at the single-step approach (see [`CConstantInterpolant`](../NN/restorators/interpolants/basic.py)). Although I set `t` and `x1` to zero to improve convergence speed, it is not necessary, as the decoder can learn to ignore them.

When we aim to produce `x0 - x1` (the direction toward `x0`), treating the current `x` as `x1`, and disregarding `t`, we obtain the standard autoregressive approach. To generate the final output, we can start from some noise and progressively move toward the target `x0` value. This description is quite abstract because there are various ways to implement it, but I hope you grasp the main idea.

Finally, we can choose to produce `x1`, either directly or indirectly, using `t` as a conditional parameter and connecting it with beta schedulers and properties of the Gaussian distribution. This corresponds to the diffusion approach. As a proof of concept, I implemented it in the [`CDDIMInterpolantSampler`](../NN/restorators/samplers/CDDIMInterpolantSampler.py) class, which can be seamlessly integrated as a replacement for the conventional DDIM/DDPM sampler.

## Diffusion

In the diffusion restorator, the decoder predicts noise that is added to the initial color and than we perform reverse diffusion to get the final output. Yeah, such a complex topic could be explained in a single sentence :)