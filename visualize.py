from Utils.utils import setupGPU, load_config, setGPUMemoryLimit
setupGPU() # call it on startup to prevent OOM errors on my machine

import tensorflow as tf
import numpy as np
import cv2, os, argparse, shutil
from NN import model_from_config
from Utils import dataset_from_config
from Utils.visualize import generateImage
from Utils.CFilesDataLoader import CFilesDataLoader
   
def makeImageProcessor(unnormalizeImg):
  def _processImage(img):
    img = unnormalizeImg(img)
    # to numpy if needed
    if tf.is_tensor(img): img = img.numpy()
    np.clip(img, 0, 1, out=img) # clamp to 0..1 range inplace

    if not(img.shape[2] == 3): # convert to RGB by duplicating the single channel
      img = np.repeat(img, 3, axis=2)

    if not(img.dtype == np.uint8): # 0..1 float -> 0..255 uint8
      img = (img * 255.0).astype(np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img
  return _processImage

def _processData(data, model, processImage):
  NB_BATCHES = len(data)
  for batchId, batch in enumerate(data):
    print(f'Batch {batchId}/{NB_BATCHES}....')
    (srcB, dstB) = batch
    upscaledB = model(srcB)
    print(f'Upscaled shape: {upscaledB.shape}')
    # all have the same first dimension
    N = len(upscaledB)
    assert (N == len(srcB)) and (N == len(dstB)), f'srcB={len(srcB)}, dstB={len(dstB)}, upscaledB={len(upscaledB)}'

    for i in range(0, N):
      yield {
        'original': processImage(dstB[i]),
        'input': processImage(srcB[i]),
        'upscaled': processImage(upscaledB[i])
      }
    continue
  return

def _data_from_dataset(config):
  datasetConfig = config['dataset']
  dataset = dataset_from_config(datasetConfig)
  data = dataset.make_dataset(datasetConfig['test'], split='test')
  return data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE), dataset

def _data_from_input(input, inputShape):
  files = []
  if os.path.isdir(input):
    # load all files from folder, filter by extension, only png and jpg... mind reading by copilot :)
    files = [os.path.join(input, f) for f in os.listdir(input) if f.endswith('.png') or f.endswith('.jpg')]
  if os.path.isfile(input):
    files = [input]

  if len(files) == 0:
    raise ValueError(f'No files found in {input}')
  
  dataloader = CFilesDataLoader(files, targetSize=inputShape[:2], srcSize=(256, 256))
  return dataloader.iterator(), dataloader
  
def main(args):
  folder = os.path.dirname(__file__)
  config = load_config(args.config, folder=folder)
  
  if args.folder:
    folder = os.path.abspath(args.folder)
    # clear or create folder
    if os.path.exists(folder): shutil.rmtree(folder)
    os.makedirs(folder)

  model = model_from_config(config['model'])
  model.load_weights(args.model)
  print('Model loaded successfully.')
  
  data, dataset = _data_from_dataset(config) if args.input is None else _data_from_input(args.input, model.get_input_shape()[1:])
  ##############################
  modelArgs = {} # inference args: size, scale, shift, reverseArgs
  generationOutputArgs = {
    'mode': 'side by side',
    'format': 'png',
    'resize': 'resize',
  }
  
  if 'visualization' in config:
    visualizationConfig = config['visualization']
    modelArgs.update(visualizationConfig['model args'])
    generationOutputArgs = visualizationConfig.get('output', generationOutputArgs)
  # override config value by command line arg
  if args.target_size:
    modelArgs['size'] = args.target_size
  
  dataIttr = _processData(
    data,
    model=lambda x: model(x, **modelArgs),
    processImage=makeImageProcessor(dataset.unnormalizeImg)
  )
  for index, data in enumerate(dataIttr):
    generateImage(
      data=data,
      folder=folder,
      index=index,
      params=generationOutputArgs
    )
    continue
  print('Done.')
  return

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Process arguments.')
  parser.add_argument(
    '--config', type=str, required=True,
    help='Path to a single config file or a multiple config files (they will be merged in order of appearance)',
    default=[], action='append', 
  )
  parser.add_argument('--model', type=str, help='Path to model weights file', required=True)
  parser.add_argument('--target-size', type=int, help='Target size (optional)')
  parser.add_argument('--input', type=str, help='Path to image file or folder (optional)', default=None)
  # misc
  parser.add_argument('--folder', type=str, help='Path to output folder (optional)', default=None)
  parser.add_argument('--gpu-memory-mb', type=int, help='GPU memory limit in Mb (optional)')
  ########################### 
  args = parser.parse_args()
  if args.gpu_memory_mb: setGPUMemoryLimit(args.gpu_memory_mb)
  main(args)
  pass