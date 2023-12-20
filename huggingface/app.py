# script to run the app in HuggingFace space
import os, argparse
import numpy as np

from Utils import ImageProcessor_from_config
from HF.UI import AppUI
from HF.Utils import toPILImage
from HF.NNHelper import modelsFrom, inference_from_config
from Utils.WandBUtils import CWBProject
import tensorflow as tf

def preprocessImage(image_processor):
  def _preprocessImage(inputImage):
    assert isinstance(inputImage, np.ndarray), f'Invalid image type: {type(inputImage)}'
    assert 3 == len(inputImage.shape), f'Invalid image shape: {inputImage.shape}'
    assert np.uint8 == inputImage.dtype, f'Invalid image dtype: {inputImage.dtype}'
    assert 3 == inputImage.shape[-1], f'Invalid image channels: {inputImage.shape[-1]}'

    input, _ = image_processor.process(inputImage[None])
    input = image_processor.range.convertBack(input).numpy()[0]
    return toPILImage(input, isBGR=False)
  return _preprocessImage

def infer(models, image_processor):
  def _processImage(modelName, inputImage, **kwargs):
    assert isinstance(inputImage, np.ndarray), f'Invalid image type: {type(inputImage)}'
    assert 3 == len(inputImage.shape), f'Invalid image shape: {inputImage.shape}'
    assert np.uint8 == inputImage.dtype, f'Invalid image dtype: {inputImage.dtype}'
    assert 3 == inputImage.shape[-1], f'Invalid image channels: {inputImage.shape[-1]}'
    # should be 64x64, because of preprocessing
    assert (64, 64) == inputImage.shape[:2], f'Invalid image shape: {inputImage.shape}'
    # inputImage has 3 channels, but its grayscale
    input = image_processor.range.convert(inputImage[None, ..., :1])

    model = models.get(modelName, None)
    assert model is not None, f'Invalid model name: {modelName}'

    upscaled = res = model(images=input, raw=inputImage, **kwargs)
    extras = {}
    if isinstance(res, dict):
      upscaled = res.pop('upscaled', None)
      extras = res

    if tf.is_tensor(upscaled):
      upscaled = image_processor.range.convertBack(upscaled).numpy()[0]

    if not(upscaled is None): # validate the output
      assert isinstance(upscaled, np.ndarray), f'Invalid image type: {type(upscaled)}'
      assert 3 == upscaled.shape[-1], f'Invalid image channels: {upscaled.shape[-1]}'
      assert 3 == len(upscaled.shape), f'Invalid image shape: {upscaled.shape}'
      upscaled = toPILImage(upscaled, isBGR=False)

    return dict(upscaled=upscaled, **extras)
  return _processImage

def run2inference(run, runName=None):
  if runName is None: runName = run.name
  runConfig = run.config
  runName = "%s (%s, loss: %.5f)" % (runName, run.id, run.bestLoss)
  # add corresponding HF info
  runConfig['huggingface'] = { "name": runName, "wandb": run.fullId }
  return list(inference_from_config(runConfig))

def SPModels(project):
  bestPerGroup = project.groups(onlyBest=True)
  bestPerGroup = {k: v for k, v in bestPerGroup.items() if k.startswith('Single-pass | ')}
  for run in bestPerGroup.values():
    for model in run2inference(run):
      yield model
    continue
  return

def main(args):
  WBProject = CWBProject('green_wizard/FranNet')
  folder = os.path.dirname(os.path.abspath(__file__))
  # load list of models from the folder "configs"
  models = modelsFrom(os.path.join(folder, 'configs'))
  # add some models from W&B
  models.extend( list(SPModels(WBProject)) )
  bestGroups = WBProject.groups(onlyBest=True)
  models.extend( run2inference(bestGroups['AR | direction']) )
  models.extend( run2inference(bestGroups['DDPM v2 | Basic']) )
  # some of "DDPM-V, encoder v2, masking-8, residual"
  models.extend( run2inference(bestGroups['DDPM-V, encoder v2, masking-8, residual, RGB']) )
  models.extend( run2inference(bestGroups['DDPM-V, encoder v2, masking-8, residual, LAB']) )
  models.extend( run2inference(bestGroups['DDPM-V, encoder v2, masking-8, residual, LAB, structured']) )

  models.extend( run2inference(bestGroups['AR direction, encoder v2, masking-8, residual, LAB, structured']) )

  # convert to dict
  models = {model.name: model for model in models}

  # Default processor used for CelebA
  celeba_processor = ImageProcessor_from_config('celeba')
  
  app = AppUI(
    preprocessImage=preprocessImage(celeba_processor),
    processImage=infer(models, celeba_processor),
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