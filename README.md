# Frankenstein Network (FranNet)

FranNet is a neural network architecture that combines a CNN encoder with a NeRF-like decoder to perform upscaling, denoising, and colorization of input images. The purpose of this project is to explore the capabilities and limitations of this approach and implement it as a proof of concept. A bit more detailed description of the network can be found in the [overview](docs/overview.md).

<img src="docs/img/scheme.png" width="70%" style="display: block; margin: auto;" />

The choice of upscaling and colorization tasks was made because they are interesting and visually appealing, and they do not require a significant amount of resources. Additionally, these tasks lend themselves well to the application of NeRF (Neural Radiance Fields). While NeRF is typically used for different purposes, I was intrigued by its ability to generate images from individual rays/points, which is ideal for upscaling.

> **DISCLAIMER/WARNING: The purpose of this project is primarily exploratory, and it is not feasible to achieve any form of photorealism due to severe resource limitations. As a result, the practical applicability of the project is currently limited. The neural network utilized in this project has been designed with fewer than 600,000 parameters, leading to a reduction in overall quality. It is important to note that all experiments were constrained to 15 training epochs (2-3 hours on Google Colab using Tesla T4 GPU).**

## Project documentation navigation

- [Current setup and dataset description](docs/current-setup.md)
- [Baselines](docs/baselines.md)
- [Visualizations and results](docs/visualizations.md)

Technical details of the project:
- [Detailed project overview](docs/overview.md)
- [Experiments](docs/experiments.md)

Additionally, there are some [Useful Links](docs/useful-links.md) to articles, videos, and other resources that were beneficial during the project.