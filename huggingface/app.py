# script to run the app in HuggingFace space
import os, argparse
import numpy as np

from Utils.CImageProcessor import CImageProcessor
from HF.UI import AppUI
from HF.Utils import toPILImage
from HF.NNHelper import modelsFrom, CNNModel
from Utils.WandBUtils import CWBProject

def preprocessImage(image_processor):
  def _preprocessImage(inputImage):
    assert isinstance(inputImage, np.ndarray), f'Invalid image type: {type(inputImage)}'
    assert 3 == len(inputImage.shape), f'Invalid image shape: {inputImage.shape}'
    assert np.uint8 == inputImage.dtype, f'Invalid image dtype: {inputImage.dtype}'
    assert 3 == inputImage.shape[-1], f'Invalid image channels: {inputImage.shape[-1]}'

    input, _ = image_processor.process(inputImage[None])
    input = image_processor.unnormalizeImg(input).numpy()[0]
    return toPILImage(input)
  return _preprocessImage

def infer(models, image_processor):
  def _processImage(modelName, inputImage, targetResolution, **kwargs):
    targetResolution = int(targetResolution)
    assert 128 <= targetResolution <= 1024, f'Invalid target resolution: {targetResolution}'

    assert isinstance(inputImage, np.ndarray), f'Invalid image type: {type(inputImage)}'
    assert 3 == len(inputImage.shape), f'Invalid image shape: {inputImage.shape}'
    assert np.uint8 == inputImage.dtype, f'Invalid image dtype: {inputImage.dtype}'
    assert 3 == inputImage.shape[-1], f'Invalid image channels: {inputImage.shape[-1]}'
    # should be 64x64, because of preprocessing
    assert (64, 64) == inputImage.shape[:2], f'Invalid image shape: {inputImage.shape}'
    # but we need to preprocess it again
    input, _ = image_processor.process(inputImage[None])

    model = models.get(modelName, None)
    assert model is not None, f'Invalid model name: {modelName}'

    upscaled = model(input, size=targetResolution, **kwargs)
    upscaled = image_processor.unnormalizeImg(upscaled).numpy()[0]
    assert 3 == upscaled.shape[-1], f'Invalid image channels: {upscaled.shape[-1]}'
    assert 3 == len(upscaled.shape), f'Invalid image shape: {upscaled.shape}'
    return {
      'upscaled': toPILImage(upscaled, isBGR=True),
    }
  return _processImage

def SPModels():
  project = CWBProject('green_wizard/FranNet')

  bestPerGroup = project.groups(onlyBest=True)
  bestPerGroup = {k: v for k, v in bestPerGroup.items() if 'single-pass' in k.lower()}
  # convert to models
  models = {}
  for runName, run in bestPerGroup.items():
    runConfig = run.config
    runName = "%s (%s, loss: %.5f)" % (runName, run.id, run.bestLoss)
    # add corresponding HF info
    runConfig['huggingface'] = {
      "name": runName,
      "wandb": run.fullId,
      "kind": "single-pass",
    }
    # add to models
    models[runName] = CNNModel(runConfig)
    continue
  return models

def main(args):
  folder = os.path.dirname(os.path.abspath(__file__))
  # load list of models from the folder "configs"
  models = modelsFrom(os.path.join(folder, 'configs'))
  # add some models from W&B
  models.update( SPModels() )

  # Processor used for CelebA
  image_processor = CImageProcessor(
    image_size=64,
    reverse_channels=False,
    to_grayscale=True,
    normalize_range=True
  )
  
  app = AppUI(
    preprocessImage=preprocessImage(image_processor),
    processImage=infer(models, image_processor),
    models=models,
  )
  app.queue() # enable queueing of requests/events
  app.launch(inline=False, server_port=args.port, server_name=args.host)
  return

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--port', type=int, default=7860)
  parser.add_argument('--host', type=str, default=None)
  args = parser.parse_args()
  main(args)