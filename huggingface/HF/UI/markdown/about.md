# About

This web UI serves as a demonstration of the capabilities of the FrankNet project, which aims to explore various aspects of artificial intelligence. It acts as a proof-of-concept and playground for experimenting with different techniques.

The primary functionality of the application involves compressing input images to a low resolution of 64x64 pixels in grayscale. The neural network's task is to then reconstruct the missing details and colors in the image. What makes this project interesting is that it achieves this goal with a small-sized neural network, less than 0.6 million parameters, and within a limited training time of less than 3 hours. **Please consider these factors when evaluating the results, as well as the fact that only basic models are currently presented.**

For more detailed information and access to the project's code, you can visit the [GitHub repository](https://github.com/GreenWizard2015/FranNet). Feel free to explore and experiment with the project to gain insights into the possibilities of artificial intelligence in image compression and reconstruction.

Workflow for working with the application:
1) Load or select an image from the examples.
2) Choose the desired resolution for the generated image.
3) Select the model type, model, and parameters.
4) Start the generation process.

There is a comparison area at the bottom of the page that allows you to compare images side by side. You can copy the image to the comparison area using the "Send to left" and "Send to right" buttons.