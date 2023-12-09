import tensorflow as tf
import Utils.colors as colors

class CImageProcessor:
  def __init__(self, image_size, to_grayscale, format, range):
    self._imageSize = image_size
    self._toGrayscale = to_grayscale
    self._range = colors.convertRanges(range, '-1..1')
    self._internalRange = colors.convertRanges(range, '0..1')
    self._outputRange = colors.convertRanges('0..1', '-1..1')

    format = format.lower()
    assert format in ['rgb', 'bgr'], 'Invalid format: %s' % format
    self._format = format
    return
  
  @property
  def range(self):
    return self._range

  def _squareCrop(self, img, args):
    s = tf.shape(img)
    B, H, W, C = s[0], s[1], s[2], s[3]
    # predefined crop size or crop to the smallest dimension
    crop_size = args.get('crop size', None)
    if crop_size is None: crop_size = tf.minimum(H, W)
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
    img = self._internalRange.convert(img)
    img = self._squareCrop(img, args=args)
    # ensure that the image is in the RGB color space
    if 'bgr' == self._format: img = img[..., ::-1] # BGR to RGB
    return img
    
  def _srcImage(self, img):
    img = tf.image.resize(img, [self._imageSize, self._imageSize])
    if self._toGrayscale:
      return tf.image.rgb_to_grayscale(img)
      
    return img
  
  def _destImage(self, img):
    return img

  def _checkInput(self, img):
    tf.assert_equal(tf.rank(img), 4)
    tf.assert_equal(tf.shape(img)[-1], 3)
    tf.debugging.assert_integer(img)
    self._internalRange.check(img)
    return

  def process(self, config_or_image):
    isConfig = isinstance(config_or_image, dict)
    args = {
      'random crop': False,
      'crop size': None
    }
    if isConfig:
      config = config_or_image
      args['random crop'] = config.get('random crop', False)
      args['crop size'] = config.get('crop size', None)
      pass

    def _process(img):
      self._checkInput(img)
      
      img = self._prepare(img, args)
      src = self._srcImage(img)
      dest = self._destImage(img)
      # NOTE: ALWAYS return images in the -1..1 range in RGB color space
      return(
        self._outputRange.convert(src),
        self._outputRange.convert(dest)
      )
    
    if isConfig: return _process
    return _process(config_or_image) # apply to image directly