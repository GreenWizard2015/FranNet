import tensorflow as tf
import tensorflow_datasets as tfds

class CCelebADataset:
  def __init__(self, image_size=64, batch_size=32):
    self._imageSize = image_size
    self._batchSize = batch_size

    self._celeb_a_builder = tfds.builder("celeb_a")
    self._celeb_a_builder.download_and_prepare()

  @staticmethod
  def normalizeImg(x): return (x * 2.0) - 1.0

  @staticmethod
  def unnormalizeImg(x): return (1.0 + x) / 2.0

  def _process(self, data):
    img = data['image']
    img = tf.cast(img, tf.float32) / 255.0

    H, W = img.shape[1], img.shape[2]
    CROP_SIZE = min(H, W)
    if not (H == CROP_SIZE):
      d = (H - CROP_SIZE) // 2
      img = img[:, d:-d, :, :]
      pass
    if not (W == CROP_SIZE):
      d = (W - CROP_SIZE) // 2
      img = img[:, :, d:-d, :]
      pass
    
    src = tf.image.rgb_to_grayscale(img[..., ::-1]) # bgr -> rgb -> grayscale
    src = tf.image.resize(src, [self._imageSize, self._imageSize])
    return(self.normalizeImg(src), self.normalizeImg(img))

  def as_dataset(self, split, batch_size=None, limit=None):
    batch_size = batch_size or self._batchSize
    res = self._celeb_a_builder.as_dataset(split=split)
    if limit: res = res.take(limit)
    if batch_size: res = res.batch(batch_size)
    return res.map(self._process)
  
  def make_dataset(self, config, split):
    res = self.as_dataset(
      split,
      batch_size=config.get('batch_size', self._batchSize),
      limit=config.get('limit', None),
    )
    if 'repeat' in config: res = res.repeat(config['repeat'])
    if 'shuffle' in config: res = res.shuffle(config['shuffle'])
    return res
  
  @property
  def input_shape(self):
    return (self._imageSize, self._imageSize, 1)