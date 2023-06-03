# Single-pass restorator visualization

Visualizations were created for the following models in decreasing order of loss:

- Basic+masking-dynamic (loss: 0.00691)
- Basic+masking-uniform (loss: 0.00671)
- Basic (loss: 0.00671)
- Basic+masking-4 (loss: 0.00651)
- Basic+masking-16 (loss: 0.00646)
- Basic+masking-32 (loss: 0.00639)
- Basic+masking-8 (loss: 0.00638)

As a reminder, as stated [here](baselines.md), the baseline loss is set at `0.01100`. All models have successfully surpassed the baseline and have been able to restore a portion of the original image's color while simultaneously upscaling it from 64x64 to 178x178 pixels.

Before proceeding, it is recommended that you read [this note on visualizations](visualize-note.md) for better understanding.

## Restoration of the entire image

![](img/visualize-single-pass/full.jpg)

It is interesting to note that models trained on larger masked areas, such as a 4x4 or 8x8 [masking grid](masking-note.md), tend to better preserve textures. This could potentially be utilized in the future to enhance the results.

## Restoration of a local region of the image

(To understand the concept of local region restoration, please refer to the following [note](visualize-note.md).)

![](img/visualize-single-pass/local.jpg)

The results are not particularly impressive, but the ability to reconstruct a local region is an interesting feature. With more accurate models, the results would be much better.