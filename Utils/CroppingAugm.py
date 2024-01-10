import tensorflow as tf

#############
# Cropping methods
def _centerSquareCrop(img, crop_size):
  s = tf.shape(img)
  B, H, W, C = s[0], s[1], s[2], s[3]
  # predefined crop size or crop to the smallest dimension
  if crop_size is None: crop_size = tf.minimum(H, W)
  sH = (H - crop_size) // 2
  sW = (W - crop_size) // 2
  
  res = img[:, sH:(sH + crop_size), sW:(sW + crop_size), :]
  tf.debugging.assert_equal(tf.shape(res), (B, crop_size, crop_size, C))
  return res

# Create a random square crop of the image
# Crops size and position are the same for all images in the batch
def _randomSharedSquareCrop(img, crop_size):
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
  return res

def _processCropSize(sz, target_crop_size):
  if sz is None: return target_crop_size
  if isinstance(sz, int): return sz
  if isinstance(sz, float):
    return tf.cast(tf.cast(target_crop_size, tf.float32) * sz, tf.int32)
  raise ValueError('Invalid crop size: %s' % sz)

# Create a random square crop of the image
def _randomSquareCrop(img, target_crop_size, minSize, maxSize):
  s = tf.shape(img)
  B, H, W, C = s[0], s[1], s[2], s[3]
  # predefined crop size or crop to the smallest dimension
  if target_crop_size is None: target_crop_size = tf.minimum(H, W)
  # preprocess crop size
  minSize = _processCropSize(minSize, target_crop_size=target_crop_size)
  maxSize = _processCropSize(maxSize, target_crop_size=target_crop_size)
  ##########################################
  def _crop(image):
    tf.debugging.assert_equal(tf.rank(image), 3)
    cropSize = tf.random.uniform((), minval=minSize, maxval=maxSize + 1, dtype=tf.int32)
    sH = tf.random.uniform((), minval=0, maxval=H - cropSize + 1, dtype=tf.int32)
    sW = tf.random.uniform((), minval=0, maxval=W - cropSize + 1, dtype=tf.int32)
    crop = image[sH:(sH + cropSize), sW:(sW + cropSize), :]
    crop = tf.image.resize(crop, [target_crop_size, target_crop_size])
    crop = tf.cast(crop, image.dtype)
    tf.debugging.assert_equal(tf.shape(crop), (target_crop_size, target_crop_size, C))
    return crop
  
  res = tf.map_fn(_crop, img)
  tf.debugging.assert_equal(tf.shape(res), (B, target_crop_size, target_crop_size, C))
  return res

#################
# TODO: Currently we didn't utilize the fact that we higher input resolution
#       ImageProcessor should be updated to use the higher resolution
#       this could be done by serving the sampled pixels from the higher resolution
#       instead of sampling them during the training
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

def _ultraGridCrop(img, target_crop_size, minSize, maxSize):
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
  def _crop(_):
    image = img
    tf.debugging.assert_equal(tf.rank(image), 3)
    cropSize = tf.random.uniform((), minval=minSize, maxval=maxSize + 1, dtype=tf.int32)
    sH = tf.random.uniform((), minval=0, maxval=H - cropSize + 1, dtype=tf.int32)
    sW = tf.random.uniform((), minval=0, maxval=W - cropSize + 1, dtype=tf.int32)
    crop = image[sH:(sH + cropSize), sW:(sW + cropSize), :]
    crop = tf.image.resize(crop, [target_crop_size, target_crop_size])
    crop = tf.cast(crop, image.dtype)
    tf.debugging.assert_equal(tf.shape(crop), (target_crop_size, target_crop_size, C))
    return crop

  res = tf.map_fn(_crop, tf.range(B), fn_output_signature=img.dtype)
  tf.debugging.assert_equal(tf.shape(res), (B, target_crop_size, target_crop_size, C))
  return res
#################
def configToCropper(config):
  crop_size = config.get('crop size', None)
  isSimpleCrop = (crop_size is None) or isinstance(crop_size, int)
  if isSimpleCrop and not config['random crop']: # simple center crop
    return lambda img: _centerSquareCrop(img, crop_size)
  # use random cropping
  if isSimpleCrop and config['shared crops']: # fast random crop
    return lambda img: _randomSharedSquareCrop(img, crop_size)
  
  # random crop with different crop size for each image
  minSize = config.get('min crop size', None)
  maxSize = config.get('max crop size', None)
  if config.get('ultra grid', False):
    if minSize is None: minSize = 0.1
    if maxSize is None: maxSize = 1.0
    return lambda img: _ultraGridCrop(img, crop_size, minSize, maxSize)
  return lambda img: _randomSquareCrop(img, crop_size, minSize, maxSize)