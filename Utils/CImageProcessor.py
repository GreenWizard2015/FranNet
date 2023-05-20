import tensorflow as tf

class CImageProcessor:
  def __init__(self, image_size, reverse_channels, to_grayscale, normalize_range):
    self._imageSize = image_size
    self._reverseChannels = reverse_channels
    self._toGrayscale = to_grayscale
    self._normalizeRange = normalize_range
    return

  def normalizeImg(self, x):
    return (x * 2.0) - 1.0

  def unnormalizeImg(self, x):
    return (1.0 + x) / 2.0

  def _prepare(self, img):
    if self._normalizeRange:
      img = tf.cast(img, tf.float32) / 255.0
    
    # TODO: add support for random cropping
    s = tf.shape(img)
    H, W = s[1], s[2]
    crop_size = tf.minimum(H, W)
    if not (H == crop_size):
      d = (H - crop_size) // 2
      img = img[:, d:-d, :, :]
    if not (W == crop_size):
      d = (W - crop_size) // 2
      img = img[:, :, d:-d, :]
    
    if self._reverseChannels: img = img[..., ::-1]
    return img
    
  def _srcImage(self, img):
    img = tf.image.resize(img, [self._imageSize, self._imageSize])
    if self._toGrayscale: img = tf.image.rgb_to_grayscale(img)
    return self.normalizeImg(img)
  
  def _destImage(self, img):
    return self.normalizeImg(img)

  def process(self, img):
    img = self._prepare(img)
    return( self._srcImage(img), self._destImage(img) )