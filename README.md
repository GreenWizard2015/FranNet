# Frankenstein Network (FranNet)

(ChatGPT generated project description)

FranNet is a neural network architecture that incorporates a CNN encoder with a NeRF-like decoder to perform upscaling, denoising, and colorization of an input image. The main purpose of this project is to explore the possibilities and limitations of this approach, and to implement it as a proof of concept.

## Main Parts of the Network

### Encoder

The encoder is a simple CNN that takes an input image (downsampled and converted to grayscale) and produces a single main latent vector and a bunch of feature maps with different resolutions. The encoder's role is to extract high-level features from the input image and produce a compressed representation that is fed into the decoder.

### Decoder

The decoder receives as input the coordinates of a point (x and y, in the range 0..1) to reconstruct, the main latent vector, and extracted from a feature map correspondent vectors. The decoder predicts noise that is added to the initial point color to produce the final output. Initially, the decoder was just an ordinary NeRF-like network that immediately outputs predicted RGB values, but later it was modified to use a diffusion approach.

## Single step Approach

In the single-step approach, the decoder produces the final output in a single step. This approach can produce images of arbitrary size and performs diffusion on each point independently, which is much more efficient.

## Diffusion Approach

In the diffusion approach, the decoder predicts noise that is added to the initial point color. This approach is more robust to noise and can produce better results than the single-step approach, but it requires more training time and memory.

## Autoregressive Approach

In the autoregressive approach, the decoder produces the final output pixel-by-pixel in a sequential manner. This approach can produce high-quality results but is slow and computationally expensive.

## Advantages of the Approach

The main advantage of FranNet is that it can perform upscaling, denoising, and colorization of an input image using a single neural network architecture. It is also able to produce images of arbitrary size and can be trained using only a small fraction of target values, which allows us to control the trade-off between performance and training speed and/or memory usage.

## Applications of the Architecture

FranNet can be used in a variety of applications, including image editing and restoration, computer vision, and graphics. It can also be used in real-time applications where speed and efficiency are crucial.

## Conclusion

FranNet is a neural network architecture that incorporates a CNN encoder with a NeRF-like decoder to perform upscaling, denoising, and colorization of an input image. It is a proof of concept and not a state-of-the-art approach, but it shows the potential and limitations of this approach. FranNet can be useful in a variety of applications and is a promising area of research for future work.