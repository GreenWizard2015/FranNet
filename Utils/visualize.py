import os
import numpy as np
import tensorflow as tf
import cv2
from Utils.CFilesDataLoader import CFilesDataLoader
from Utils import dataset_from_config

def generateImage_sideBySide(
  images, titles, margin,
  textHeight=0.05, textMargin=10, textOpacity=0.5, textThickness=3, textFont=cv2.FONT_HERSHEY_SIMPLEX
):
  assert(len(images) == len(titles))
  size = images[0].shape[:2]
  assert(all([img.shape[:2] == size for img in images]))

  width = (size[1] * len(images)) + (margin * (len(images) - 1))
  height = (size[0] * 1) + (margin * 0) # 1 row
  lineHeight = max((12, int(height * textHeight)))

  x, y = 0, 0
  img = np.zeros((height, width, 3), dtype=np.uint8)
  for i in range(len(images)):
    h, w = images[i].shape[:2]
    img[y:y+h, x:x+w] = images[i]
    # draw title. red, at the bottom, centered
    textSz, baseline = cv2.getTextSize(titles[i], textFont, 1, textThickness)
    textSz = (textSz[0], textSz[1] + baseline)
    scale = lineHeight / textSz[1]
    tw, th = (int(textSz[0] * scale), int(textSz[1] * scale))
    tx, ty = pos = (x + (w // 2) - (tw // 2), y + h - (th + margin))
    # draw text background. black, opacity
    rect = {
      'x1': tx - textMargin,
      'y1': ty - th - textMargin,
      'x2': tx + tw + textMargin,
      'y2': ty + textMargin
    }
    img[rect['y1']:rect['y2'], rect['x1']:rect['x2']] = cv2.addWeighted(
      img[rect['y1']:rect['y2'], rect['x1']:rect['x2']],
      1.0 - textOpacity,
      np.zeros((rect['y2'] - rect['y1'], rect['x2'] - rect['x1'], 3), dtype=np.uint8),
      textOpacity,
      0
    )
    # draw text
    cv2.putText(img, titles[i], pos, textFont, scale, (0, 0, 255), textThickness)

    x += w + margin
    continue
  return img

def _generateImage_resize(data, params):
  biggestSize = max([v.shape[0] for v in data.values()])
  resizeFun = None
  if 'resize' == params:
    resizeFun = lambda x: cv2.resize(x, (biggestSize, biggestSize), interpolation=cv2.INTER_LANCZOS4)

  if 'centered' == params:
    def apply(x):
      # pad x to biggestSize
      H, W = x.shape[:2]
      padH = biggestSize - H
      padW = biggestSize - W
      x = np.pad(x, ((padH//2, padH - padH//2), (padW//2, padW - padW//2), (0, 0)), 'constant', constant_values=255)
      return x
    resizeFun = apply

  assert resizeFun is not None, 'Something went wrong with the resize function.'
  return {k: resizeFun(v) for k, v in data.items()}

def generateImage(data, folder, index, params):
  format = params['format']
  sizes = {k: v.shape[0] for k, v in data.items()}
  titles = [
    'GT (x%.2f)' % (sizes['original'] / sizes['original']),
    'Input (x%.2f)' % (sizes['input'] / sizes['original']),
    'Upscaled (x%.2f)' % (sizes['upscaled'] / sizes['original']),
  ]

  if 'resize' in params:
    data = _generateImage_resize(data, params['resize'])

  mode = params['mode']
  if 'side by side' == mode:
    img = generateImage_sideBySide([data['original'], data['input'], data['upscaled']], titles, margin=10)
    cv2.imwrite(os.path.join(folder, f'{index:04d}.{format}'), img)
    return
  
  if 'separate' == mode:
    for k, v in data.items():
      if params.get(k, True):
        cv2.imwrite(os.path.join(folder, f'{index:04d}_{k}.{format}'), v)
    return
  
  raise ValueError(f'Unknown mode: {mode}')

def data_from_dataset(config):
  datasetConfig = config['dataset']
  dataset = dataset_from_config(datasetConfig)
  data = dataset.make_dataset(datasetConfig['test'], split='test')
  return data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE), dataset

def data_from_input(input, inputShape):
  files = []
  if os.path.isdir(input):
    # load all files from folder, filter by extension, only png and jpg
    files = [os.path.join(input, f) for f in os.listdir(input) if f.endswith('.png') or f.endswith('.jpg')]
  if os.path.isfile(input):
    files = [input]

  if len(files) == 0:
    raise ValueError(f'No files found in {input}')
  
  dataloader = CFilesDataLoader(files, targetSize=inputShape[:2], srcSize=(256, 256))
  return dataloader.iterator(), dataloader

def withText(
  img, text,
  position='top',
  color=(0, 0, 255), background=(255, 255, 255),
  thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX, scale=1.0,
  margin=10
):
  assert position in ['top'], 'Unknown position: %s' % position
  assert len(img.shape) == 3 and img.shape[2] == 3, 'Image must be 3-channel RGB'
  assert len(color) == 3, 'Color must be 3-channel RGB'
  assert len(background) == 3, 'Background must be 3-channel RGB'

  # calculate text size
  textSz, baseline = cv2.getTextSize(text, font, scale, thickness)
  textSz = (textSz[0], textSz[1] + baseline)

  # add space for text
  img = cv2.copyMakeBorder(img, textSz[1] + margin * 2, 0, 0, 0, cv2.BORDER_CONSTANT, value=background)
  # calculate text position
  width = img.shape[1]

  # draw text at center top
  pos = (width // 2 - textSz[0] // 2, textSz[1] + margin)
  cv2.putText(img, text, pos, font, scale, color, thickness)
  return img

def makeGrid(images, columns):
  rows = (len(images) + columns - 1) // columns
  gridRows = []
  for i in range(rows):
    chunk = images[i * columns : (i + 1) * columns]
    row = np.concatenate(chunk, axis=1)
    gridRows.append(row)
    continue
  # pad last row to make it the same size as others
  if len(gridRows) > 1:
    lastRow, firstRow = gridRows[-1], gridRows[0]
    if not (lastRow.shape == firstRow.shape):
      padding = np.subtract(firstRow.shape, lastRow.shape)
      padding = tuple([(0, x) for x in padding])
      gridRows[-1] = np.pad(lastRow, padding, mode='constant', constant_values=255)
  # concatenate rows into a grid
  grid = np.concatenate(gridRows, axis=0)
  return grid

# returns a function that processes an image
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