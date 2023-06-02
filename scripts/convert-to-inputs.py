# this script converts the data from the file/folder/dataset to the input format for the model
import argparse, os, sys, re
# add the root folder of the project to the path
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../'))

from Utils.utils import setupGPU, load_config, merge_configs, JSONHelper
setupGPU() # call it on startup to prevent OOM errors on my machine

import cv2, os, argparse, shutil
from Utils.visualize import generateImage, data_from_dataset, data_from_input, makeImageProcessor

def _processData(data, processImage):
  NB_BATCHES = len(data)
  for batchId, batch in enumerate(data):
    print(f'Batch {batchId}/{NB_BATCHES}....')
    (srcB, dstB) = batch
    for i in range(len(srcB)):
      yield {
        'original': processImage(dstB[i]),
        'input': processImage(srcB[i]),
      }
    continue
  return

def datasetFrom(args, config):
  if args.input is None:
    return lambda input_shape: data_from_dataset(config)
  # otherwise from input
  return lambda input_shape: data_from_input(args.input, input_shape)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Process arguments.')
  parser.add_argument(
    '--config', type=str, required=True,
    help='Path to a single config file or a multiple config files (they will be merged in order of appearance)',
    default=[], action='append', 
  )
  parser.add_argument('--input', type=str, help='Path to image file or folder (optional)', default=None)
  parser.add_argument('--model-input-shape', type=str, help='Model input shape (optional)', default='(64, 64, 1)')
  parser.add_argument('--target-size', type=int, help='Target size (optional)', default=None)
  # misc
  parser.add_argument('--folder', type=str, help='Path to output folder (optional)', default=None)
  parser.add_argument('--format', type=str, help='Output format (optional)', default='png')
  ########################### 
  args = parser.parse_args()
  folder = os.getcwd()
  if args.folder:
    folder = os.path.abspath(args.folder)
    # clear/create folder
    if os.path.exists(folder): shutil.rmtree(folder)
    os.makedirs(folder)
    pass

  config = load_config(args.config, folder=os.getcwd())
  # should be specified input flag or config contains 'dataset' section
  assert (args.input is not None) or ('dataset' in config), 'either input or dataset section in config is required'

  datasetProvider = datasetFrom(args, config)
  
  if not os.path.exists(folder): os.makedirs(folder)
  # parse a tuple of ints from string using regex. Example: '(64, 64, 1)' -> (64, 64, 1)
  pattern = re.compile(r'\((\d+),\s*(\d+),\s*(\d+)\)')
  modelInputShape = tuple(map(int, pattern.match(args.model_input_shape).groups()))
  print(f'Using model input shape: {modelInputShape}')

  data, dataset = datasetProvider(modelInputShape)
  dataIttr = _processData(
    data,
    processImage=makeImageProcessor(dataset.unnormalizeImg)
  )
  for index, data in enumerate(dataIttr):
    inputImg = data['input']
    if args.target_size is not None:
      inputImg = cv2.resize(inputImg, (args.target_size, args.target_size))
      
    cv2.imwrite(os.path.join(folder, f'{index}.{args.format}'), inputImg)
    continue
  print('Done.')
  pass