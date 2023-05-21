import argparse, os, sys
# add the root folder of the project to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Utils.utils import setupGPU, load_config, masking_from_config
setupGPU() # call it on startup to prevent OOM errors on my machine

import cv2
import numpy as np
from Utils import dataset_from_config

def main(args):
  columns = args.columns
  rows = args.rows
  output = args.output
  folder = os.path.dirname(__file__)
  config = load_config(args.config, folder=folder)
  # Select dataset
  dataset = dataset_from_config(config['dataset'])
  test_data = dataset.make_dataset(config['dataset']['test'], split='test')

  # take a sample from the dataset
  for src, _ in test_data.take(1):
    src = src[:1]
    break
  print(f'Source shape: {src.shape}')

  # take config of masking for train dataset
  masking = masking_from_config(config['dataset']['train']['masking'])
  # collect images
  images = [dataset.unnormalizeImg(src)[0].numpy()]
  while len(images) < columns * rows:
    augmented, _ = masking(src, src)
    augmented = dataset.unnormalizeImg(augmented)[0].numpy()
    images.append(augmented)
    continue
  images = images[:columns * rows]

  # create grid
  gridRows = []
  for i in range(rows):
    row = []
    for j in range(columns):
      row.append(images[i * columns + j])
      continue
    gridRows.append(np.concatenate(row, axis=1))
    continue
  grid = np.concatenate(gridRows, axis=0)
  print(f'Grid shape: {grid.shape}')
  # convert to uint8
  grid = (grid * 255).astype(np.uint8)
  cv2.imwrite(output, grid)
  return

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Process arguments.')
  parser.add_argument(
    '--config', type=str, required=True,
    help='Path to a single config file or a multiple config files (they will be merged in order of appearance)',
    default=[], action='append', 
  )
  # add columns and rows arguments
  parser.add_argument('--columns', type=int, required=True, help='Number of columns in the grid')
  parser.add_argument('--rows', type=int, required=True, help='Number of rows in the grid')
  # add output argument
  parser.add_argument('--output', type=str, required=True, help='Path to the output file')

  args = parser.parse_args()
  # used for creating a grid of images
  # args.config = [
  #   'configs/basic.json',
  #   'configs/experiments/masking.json',
  # ]
  # args.columns = 8
  # args.rows = 8
  # args.output = 'docs/img/masking-grid.jpg'
  main(args)