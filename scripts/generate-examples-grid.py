import argparse, os, sys
# add the root folder of the project to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Utils.utils import setupGPU, load_config
setupGPU() # call it on startup to prevent OOM errors on my machine

import cv2
import numpy as np
from Utils import dataset_from_config

def toSample(src, dest):
  H = max(src.shape[0], dest.shape[0])
  W = max(src.shape[1], dest.shape[1])

  # src could be grayscale
  if (len(src.shape) == 2) or (src.shape[2] == 1):
    src = np.repeat(src, 3, axis=2)

  # pad images and make them centered
  hP = H - src.shape[0]
  wP = W - src.shape[1]
  padH = hP // 2
  padW = wP // 2
  src = np.pad(src, ((padH, hP - padH), (padW, wP - padW), (0, 0)), mode='constant', constant_values=1)
  # dest = np.pad(dest, ((0, H - dest.shape[0]), (0, W - dest.shape[1]), (0, 0)), mode='constant', constant_values=1)

  res = np.concatenate((src, dest), axis=1)
  # convert to uint8
  res = (res * 255).astype(np.uint8)
  # black border
  res = np.pad(res, ((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
  return res

def main(args):
  folder = os.path.dirname(__file__)
  config = load_config(args.config, folder=folder)
  # Select dataset
  dataset = dataset_from_config(config['dataset'])
  test_data = dataset.make_dataset(config['dataset']['test'], split='test')
  
  columns = args.columns
  rows = args.rows
  output = args.output

  # collect images
  images = []
  for (src, dest) in test_data:
    src = dataset.unnormalizeImg(src).numpy()
    dest = dataset.unnormalizeImg(dest).numpy()
    for i in range(src.shape[0]):
      images.append(toSample(src[i], dest[i]))
    if columns * rows <= len(images): break
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
  # args.config = 'configs/basic.json'
  # args.columns = 3
  # args.rows = 3
  # args.output = 'img/examples-grid.jpg'
  main(args)