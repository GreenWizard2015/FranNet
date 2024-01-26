import tensorflow as tf
from Utils.utils import CFakeObject
from Utils.PositionsSampler import PositionsSampler
from NN.utils import extractInterpolated

def CropsProcessor(F, signature):
  return CFakeObject(F=F, signature=signature)

def _resizeTo(src_size):
  def _F(dest_size=None):
    def _FF(img):
      dest = src = img = tf.cast(img, tf.float32)
      if src_size is not None: src = tf.image.resize(img, [src_size, src_size])
      if dest_size is not None: dest = tf.image.resize(img, [dest_size, dest_size])
      return dict(src=src, dest=dest)
    return _FF
  return _F

def RawProcessor(src_size):
  return CropsProcessor(
    _resizeTo(src_size),
    dict(src=tf.float32, dest=tf.float32)
  )

def SubsampleProcessor(target_crop_size, N, sampler='uniform'):
  assert isinstance(N, int), 'Invalid N: %s' % N
  assert N > 0, 'Invalid N: %s' % N
  sampler = PositionsSampler(sampler)
  resizer = _resizeTo(target_crop_size)(None)
  def _F(dest_size=None): # dest_size is ignored
    def _FF(img):
      img = tf.cast(img, tf.float32)
      positions = sampler((1, N, 2))
      sampled = extractInterpolated(img[None], positions)
      src = resizer(img)['src']
      return dict(src=src, sampled=sampled[0], positions=positions[0])
    return _FF
  
  return CropsProcessor(
    F=_F,
    signature=dict(src=tf.float32, sampled=tf.float32, positions=tf.float32)
  )
#############
# Cropping methods
def _centerSquareCrop(img, crop_size, processor):
  s = tf.shape(img)
  B, H, W, C = s[0], s[1], s[2], s[3]
  # predefined crop size or crop to the smallest dimension
  if crop_size is None: crop_size = tf.minimum(H, W)
  sH = (H - crop_size) // 2
  sW = (W - crop_size) // 2
  
  res = img[:, sH:(sH + crop_size), sW:(sW + crop_size), :]
  tf.debugging.assert_equal(tf.shape(res), (B, crop_size, crop_size, C))
  return tf.map_fn(processor.F(crop_size), res, fn_output_signature=processor.signature)

# Create a random square crop of the image
# Crops size and position are the same for all images in the batch
def _randomSharedSquareCrop(img, crop_size, processor):
  s = tf.shape(img)
  B, H, W, C = s[0], s[1], s[2], s[3]
  # predefined crop size or crop to the smallest dimension
  if crop_size is None: crop_size = tf.minimum(H, W)
  sH = dH = (H - crop_size) // 2
  sW = dW = (W - crop_size) // 2
  sH = tf.random.uniform((), minval=0, maxval=2*dH + 1, dtype=tf.int32)
  sW = tf.random.uniform((), minval=0, maxval=2*dW + 1, dtype=tf.int32)
  res = img[:, sH:(sH + crop_size), sW:(sW + crop_size), :]
  tf.debugging.assert_equal(tf.shape(res), (B, crop_size, crop_size, C))
  return tf.map_fn(processor.F(crop_size), res, fn_output_signature=processor.signature)

def _processCropSize(sz, target_crop_size):
  if sz is None: return target_crop_size
  if isinstance(sz, int): return sz
  if isinstance(sz, float):
    return tf.cast(tf.cast(target_crop_size, tf.float32) * sz, tf.int32)
  raise ValueError('Invalid crop size: %s' % sz)

def _extractRandomCrop(image, minSize, maxSize):
  tf.debugging.assert_equal(tf.rank(image), 3)
  H, W = tf.shape(image)[0], tf.shape(image)[1]
  cropSize = tf.random.uniform((), minval=minSize, maxval=maxSize + 1, dtype=tf.int32)
  sH = tf.random.uniform((), minval=0, maxval=H - cropSize + 1, dtype=tf.int32)
  sW = tf.random.uniform((), minval=0, maxval=W - cropSize + 1, dtype=tf.int32)
  res = image[sH:(sH + cropSize), sW:(sW + cropSize), :]
  tf.assert_equal(tf.shape(res)[:2], [cropSize, cropSize])
  return res

# Create a random square crop of the image
def _randomSquareCrop(img, target_crop_size, minSize, maxSize, processor):
  s = tf.shape(img)
  B, H, W, C = s[0], s[1], s[2], s[3]
  # predefined crop size or crop to the smallest dimension
  if target_crop_size is None: target_crop_size = tf.minimum(H, W)
  # preprocess crop size
  minSize = _processCropSize(minSize, target_crop_size=target_crop_size)
  maxSize = _processCropSize(maxSize, target_crop_size=target_crop_size)
  ##########################################
  F = processor.F(target_crop_size)
  def _crop(image):
    crop = _extractRandomCrop(image, minSize, maxSize)
    return F(crop)
  
  return tf.map_fn(_crop, img, fn_output_signature=processor.signature)

#################
# Ultra grid cropping
# Its creates a huge combined image and then crops it
def _createUltraGrid(img):
  s = tf.shape(img)
  B, H, W, C = s[0], s[1], s[2], s[3]
  # rows and columns
  N = tf.cast(tf.math.ceil(tf.math.sqrt(tf.cast(B, tf.float32))), tf.int32)
  # pad the image to the size of the grid
  img = tf.concat([img, img], axis=0)[:N*N]
  # combine images in rows
  img = tf.reshape(img, [N, N, H, W, C])
  # combine images in columns
  img = tf.transpose(img, [0, 2, 1, 3, 4])
  img = tf.reshape(img, [N * H, N * W, C])
  return img

def _ultraGridCrop(img, target_crop_size, minSize, maxSize, processor):
  s = tf.shape(img)
  B, H, W, C = s[0], s[1], s[2], s[3]
  # predefined crop size or crop to the smallest dimension
  if target_crop_size is None: target_crop_size = tf.minimum(H, W)
  ##########################################
  img = _createUltraGrid(img)
  H = tf.shape(img)[0]
  W = tf.shape(img)[1]
  newCropSize = tf.minimum(H, W)
  minSize = _processCropSize(minSize, target_crop_size=newCropSize)
  maxSize = _processCropSize(maxSize, target_crop_size=newCropSize)
  ##########################################
  F = processor.F(target_crop_size)
  def _crop(_):
    crop = _extractRandomCrop(img, minSize, maxSize)
    return F(crop)

  return tf.map_fn(_crop, tf.range(B), fn_output_signature=processor.signature)
#################
def _configToCropProcessor(config, dest_size):
  if not config.get('subsample', False):
    return RawProcessor(dest_size)

  subsample = config['subsample']
  assert isinstance(subsample, dict), 'Invalid subsample config'
  N = subsample['N']
  sampling = subsample.get('sampling', 'uniform')
  return SubsampleProcessor(dest_size, N, sampling)

def configToCropper(config, dest_size):
  assert isinstance(dest_size, int), 'Invalid dest_size: %s' % dest_size
  crop_size = config.get('crop size', None)
  isSimpleCrop = (crop_size is None) or isinstance(crop_size, int)
  cropProcessor = _configToCropProcessor(config, dest_size)
  #################
  if isSimpleCrop and not config['random crop']: # simple center crop
    return lambda img: _centerSquareCrop(img, crop_size, cropProcessor)
  # use random cropping
  if isSimpleCrop and config['shared crops']: # fast random crop
    return lambda img: _randomSharedSquareCrop(img, crop_size, cropProcessor)
  
  # random crop with different crop size for each image
  minSize = config.get('min crop size', None)
  maxSize = config.get('max crop size', None)
  if config.get('ultra grid', False):
    if minSize is None: minSize = 0.1
    if maxSize is None: maxSize = 1.0
    return lambda img: _ultraGridCrop(img, crop_size, minSize, maxSize, cropProcessor)
  return lambda img: _randomSquareCrop(img, crop_size, minSize, maxSize, cropProcessor)