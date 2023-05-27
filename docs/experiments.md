# Experiments

In the [README.md](../README.md) file, you can find a brief overview of the current setup, dataset description, and baselines.

> **DISCLAIMER/WARNING: The purpose of this project is primarily exploratory, and it is not feasible to achieve any form of photorealism due to severe resource limitations. As a result, the practical applicability of the project is currently limited. The neural network utilized in this project has been designed with fewer than 600,000 parameters, leading to a reduction in overall quality. It is important to note that all experiments were constrained to 15 training epochs (2-3 hours on Google Colab using Tesla T4 GPU).**

The [configs/experiments](../configs/experiments) folder contains the configuration files for the experiments, along with some optional customizations for them. You can start the training process for an experiment using the following command:
```
python3 train.py --config configs/basic.json --config configs/experiments/[config name 1] --config configs/experiments/[config name N]
```

More about the command-line arguments can be found in the [scripts.md](scripts.md) file.

Folder structure and brief description of some of the currently available experiments:

- [single-pass](../configs/experiments/single-pass) folder contains the experiments with single-pass restorator, which is the simplest one.
  - [single-pass/baseline.json](../configs/experiments/single-pass/baseline.json).
- [autoregressive](../configs/experiments/autoregressive) folder contains the experiments with autoregressive restorator.
  - [autoregressive/direction.json](../configs/experiments/autoregressive/direction.json) - in this experiment, the `decoder` is trained to predict the direction from the current color towards correct one. During the inference, the `sampler` starts from the random color and then iteratively restores the original color.
  - [autoregressive/ar-ddim-extended.json](../configs/experiments/autoregressive/ar-ddim-extended.json) - this is tweaked version of the diffusion approach implemented in terms of autoregressive restorator and interpolants.
  - [autoregressive/ar-ddim-extended-V.json](../configs/experiments/autoregressive/ar-ddim-extended-V.json) - same as above, but with the V-objective (predicting the mix of the initial noise and the correct value).
- [diffusion](../configs/experiments/diffusion) folder contains the experiments with diffusion restorator.
  - [diffusion/ddpm.json](../configs/experiments/diffusion/ddpm.json) - diffusion restorator that uses DDPM sampler.
  - [diffusion/ddim.json](../configs/experiments/diffusion/ddim.json) - exactly the same as above, but restorator uses the DDIM sampler instead of the DDPM one. This sampler skips the steps of the diffusion process, so it is faster.

In the root of the [configs/experiments](../configs/experiments) folder, there are some common customizations for the experiments:

- [complex-encoder.json](../configs/experiments/complex-encoder.json) - This config file contains customizations for the encoder. It uses a more complex encoder architecture that provides more flexibility for feature extraction from the input image.
- [masking.json](../configs/experiments/masking.json) - Enables the masking of the input image during the training process. The input image is split into 16x16 grid of patches, and then random masking is applied to up to 75% of the patches. The neural network should not only restore the color of the image but also the masked patches. You can see the example of the masked image [here](img/masking-grid.jpg). (other `masking-*.json` are the same, but with different grid sizes and other parameters)
- [sd-halton.json](../configs/experiments/sd-halton.json) -  customize the way noise is sampled during the training process. In some cases, quasi-random sequences can be used as a replacement for normally distributed noise. The specified range is `-4` to `4`, which corresponds to 4 sigmas of the normal distribution.
- [sd-resampled.json](../configs/experiments/sd-resampled.json) - This config file is similar to the previous one, but it uses resampled noise instead of quasi-random noise.

## Reports, results, and todo-list

> **NOTE**: please be aware that the training process involves a significant level of sparsity, which leads to very noisy metrics. When it was reasonable, I tried to run the training process multiple times and then averaged the results. However, in some cases, I had to use the results from a single run.

Models to be trained:

- [ ] Single-pass restorator
  - [x] Basic
  - [x] With masking (up to 75% of patches are masked)
    - [x] 16x16 grid
    - [x] 8x8 grid
    - [x] 4x4 grid
    - [x] Other grid sizes
  - [ ] With complex encoder
- [ ] Diffusion restorator
  - [ ] DDPM sampler (save each epoch to cherry-pick the best one later for DDIM sampler)
    - [ ] Basic
    - [ ] With halton quasi-random noise
    - [ ] With loss weighting
  - [ ] DDIM sampler
- [ ] Autoregressive restorator
  - [ ] Direction
  - [ ] DDIM extended
  - [ ] DDIM extended with V-objective

Studies to be conducted:

- [x] [Compare with and without masking](masking-ablation.md)
- [ ] Compare with and without complex encoder
- [ ] Compare different parameters for DDIM sampler, compare with DDPM
- [ ] Compare different parameters for autoregressive "direction" restorator sampler
- [ ] Show that ordinary DDIM and autoregressive DDIM are the same, in terms of inference and training
- [ ] Visualize the trajectories of the color values during the sampling process
- [ ] Compare different model sizes (**600k**, x4, x16?)
- [ ] Compare different noise sampling methods (**normal**, halton, resampled)
- [ ] Try to use quasi-random generator to sample points for training (**uniform**, halton)
- [ ] Try incorporating additional information into the input image (e.g., applying edge detection and adding the result as an additional channel).
- [ ] Compare inference time for different models