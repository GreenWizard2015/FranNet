# On visualizations and comparing models

Visualizations were generated for the following images:

![](img/testing-grid.jpg)

(Full size images can be found in the [img/testing](img/testing) folder.)

These images are clearly out-of-distribution but maintain the same semantic meaning as the training data. Additionally, it is important for the images to contain textures; otherwise, models may struggle to reconstruct colors accurately. Each image is initially downscaled to 64x64 and converted to grayscale. Then, a grid of points of the desired size (1024x1024 in this case, which is 16 times larger than the input image) is generated. The model predicts colors for each point independently, without considering the context of other points. This means that the model cannot take into account the image's overall context. The resulting colors of the points are assembled into an image of the desired size.

Below, you can see the original images used as inputs for the models. They are of the same size and in grayscale, allowing you to comprehend the challenges encountered by the models when attempting to generate colorized and upscaled images.

![](img/testing-extra/1-small.png)

![](img/testing-extra/2-small.png)

In all the visualizations, the target size is set at 1024x1024. To facilitate easier viewing, the displayed images have been reduced to 256x256 for space-saving purposes. This reduction does not significantly distort the results and enables a clear comparison between the models.