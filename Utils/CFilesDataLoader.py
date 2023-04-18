import tensorflow as tf
import numpy as np
import cv2

# TODO: add support for batch size > 1
class CFilesDataLoader:
  def __init__(self, files, targetSize, srcSize, batch_size=1, isRGB=False, normalize=True):
    self._isRGB = isRGB
    self._normalize = normalize
    self._files = files
    self._targetSize = targetSize
    self._srcSize = srcSize
    self._batch_size = batch_size
    assert (batch_size == 1), 'batch size > 1 not supported yet'
    return

  @staticmethod
  def normalizeImg(x): return (x * 2.0) - 1.0

  @staticmethod
  def unnormalizeImg(x): return (1.0 + x) / 2.0

  def _prepareImage(self, img_path):
    img = cv2.imread(img_path)
    assert not(img is None), f'Failed to load image {img_path}'
    assert img.ndim == 3, f'Image {img_path} has invalid number of dimensions: {img.ndim}'

    # TODO: find more elegant way to do this
    needNormalize = self._normalize or (np.float32 != img.dtype) or (1.0 < np.max(img))
    if needNormalize:
      img = img.astype(np.float32) / 255.0 # normalize to 0..1 range

    # TODO: find more elegant way to detect if image is RGB or BGR
    if not self._isRGB:
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    H, W = img.shape[:2]
    CROP_SIZE = min(H, W)
    if not (H == CROP_SIZE):
      d = (H - CROP_SIZE) // 2
      img = img[d:-d, :, :]
      pass
    if not (W == CROP_SIZE):
      d = (W - CROP_SIZE) // 2
      img = img[:, d:-d, :]
      pass

    img = tf.image.resize(img, [self._srcSize[0], self._srcSize[1]])
    return img
  
  def _process(self, img):
    src = tf.image.rgb_to_grayscale(img)
    src = tf.image.resize(src, [self._targetSize[0], self._targetSize[1]])
    
    return (self.normalizeImg(src)[None], self.normalizeImg(img)[None])

  def iterator(self): return self

  def __iter__(self):
    for img_path in self._files:
      yield self._process( self._prepareImage(img_path) )
    return
  
  def __len__(self): return len(self._files)