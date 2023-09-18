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

  def _squareCrop(self, img, args):
    s = tf.shape(img)
    B, H, W, C = s[0], s[1], s[2], s[3]
    # predefined crop size or crop to the smallest dimension
    crop_size = args.get('crop size', tf.minimum(H, W))
    sH = dH = (H - crop_size) // 2
    sW = dW = (W - crop_size) // 2
    if args['random crop']: # same crop for all images in the batch
      sH = tf.random.uniform((), minval=0, maxval=2*dH + 1, dtype=tf.int32)
      sW = tf.random.uniform((), minval=0, maxval=2*dW + 1, dtype=tf.int32)
      pass
    res = img[:, sH:(sH + crop_size), sW:(sW + crop_size), :]
    tf.debugging.assert_equal(tf.shape(res), (B, crop_size, crop_size, C))
    return res
               
  def _prepare(self, img, args):
    img = tf.cast(img, tf.float32)
    if self._normalizeRange: img = img / 255.0
    img = self._squareCrop(img, args=args)
    # NOTE: its a bug, should be in _srcImage,
    # but some models were trained with the channels reversed, so keep it
    if self._reverseChannels: img = img[..., ::-1]
    return img
    
  def _srcImage(self, img):
    img = tf.image.resize(img, [self._imageSize, self._imageSize])
    if self._toGrayscale: img = tf.image.rgb_to_grayscale(img)
    return self.normalizeImg(img)
  
  def _destImage(self, img):
    return self.normalizeImg(img)

  def process(self, config):
    args = {}
    args['random crop'] = config.get('random crop', False)
    if config.get('crop size', None):
      args['crop size'] = config['crop size']

    def _process(img):
      img = self._prepare(img, args)
      return( self._srcImage(img), self._destImage(img) )
    return _process