from Utils.utils import setupGPU
setupGPU()

from Utils.celeba import CCelebADataset
from NN import celebaModel
import NN.diffusion as D
import tensorflow as tf
import numpy as np
import cv2, os, argparse

def generateImage(
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

def makeProcessImage(unnormalizeImg):
  def _processImage(img, dstSize):
    img = unnormalizeImg(img)
    # to numpy if needed
    if tf.is_tensor(img): img = img.numpy()
    np.clip(img, 0, 1, out=img) # clamp to 0..1 range inplace

    if not(img.shape[2] == 3): # convert to RGB by duplicating the single channel
      img = np.repeat(img, 3, axis=2)

    if not(img.shape[:2] == dstSize):
      img = cv2.resize(img, dstSize, interpolation=cv2.INTER_NEAREST)

    if not(img.dtype == np.uint8): # 0..1 float -> 0..255 uint8
      img = (img * 255.0).astype(np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img
  return _processImage

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', type=str, required=True)
  parser.add_argument('--batch-size', type=int, default=128)
  parser.add_argument('--limit', type=int, default=None)
  parser.add_argument('--split', type=str, default='test')
  parser.add_argument('--folder', type=str, default='visualize')

  parser.add_argument('--image-size', type=int, default=64)
  # target image size
  parser.add_argument('--target-size', type=int, default=None)
  args = parser.parse_args()

  SRC_IMG_SIZE = args.image_size
  dataset = CCelebADataset(image_size=SRC_IMG_SIZE)

  model = celebaModel(SRC_IMG_SIZE)
  model.load_weights(args.model)

  if not(os.path.exists(args.folder)):
    os.makedirs(args.folder)
    
  data = dataset.as_dataset(split=args.split, limit=args.limit, batch_size=args.batch_size)
  data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  NB_BATCHES = len(data)
  dstSize = (args.target_size, args.target_size) if args.target_size else None
  processImage = makeProcessImage(dataset.unnormalizeImg)
  for batchId, batch in enumerate(data):
    print(f'Batch {batchId}/{NB_BATCHES}....')
    (srcB, dstB) = batch
    assert (srcB.shape[1] == SRC_IMG_SIZE) and (srcB.shape[2] == SRC_IMG_SIZE), f'Invalid source image size: {srcB.shape}'
    assert (dstB.shape[1] == dstB.shape[2]), f'Invalid destination image size: {dstB.shape}'

    if dstSize is None: # get destination size from the first batch
      dstSize = dstB.shape[1:3]
      print(f'Destination size: {dstSize}')

    upscaledB = model(srcB, size=dstSize[0]).numpy()
    print(f'Upscaled shape: {upscaledB.shape}')
    
    for i in range(len(srcB)):
      src = processImage(srcB[i], dstSize)
      dst = processImage(dstB[i], dstSize)
      upscaled = processImage(upscaledB[i], dstSize)

      # combine images side by side with 10px margin
      img = generateImage(
        images=[dst, src, upscaled],
        titles=[
          'GT (x1)',
          'Input (x%.1f, grayscale)' % (srcB.shape[1] / dstB.shape[1]),
          'Upscaled (x%.1f)' % (upscaledB.shape[1] / dstB.shape[1])
        ],
        margin=10
      )
      cv2.imwrite(f'{args.folder}/{batchId}_{i}.png', img)
      continue
    continue
  print('Done.')
  return

if __name__ == '__main__':
  main()
