import tensorflow as tf
import Utils.colors as colors
import Utils.CroppingAugm as CroppingAugm

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
  def range(self): return self._range

  def _prepare(self, img):
    img = self._internalRange.convert(img)
    # ensure that the image is in the RGB color space
    if 'bgr' == self._format: img = img[..., ::-1] # BGR to RGB
    return img
    
  def _srcImage(self, img):
    img = tf.image.resize(img, [self._imageSize, self._imageSize])
    if self._toGrayscale:
      return tf.image.rgb_to_grayscale(img)
      
    return img
  
  def _destImage(self, img): return img

  def _checkInput(self, img):
    tf.assert_equal(tf.rank(img), 4)
    tf.assert_equal(tf.shape(img)[-1], 3)
    tf.debugging.assert_integer(img)
    self._internalRange.check(img)
    return

  def process(self, config_or_image):
    isConfig = isinstance(config_or_image, dict)
    args = { # default arguments
      'random crop': False,
      'shared crops': True,
      'crop size': None
    }
    if isConfig:
      args = dict(args, **config_or_image)

    cropper = CroppingAugm.configToCropper(args)
    def _process(img):
      self._checkInput(img)
      
      img = cropper(img) # crop BEFORE preprocessing
      img = self._prepare(img)
      src = self._srcImage(img)
      dest = self._destImage(img)
      # NOTE: ALWAYS return images in the -1..1 range in RGB color space
      return(
        self._outputRange.convert(src),
        self._outputRange.convert(dest)
      )
    
    if isConfig: return _process
    return _process(config_or_image) # apply to image directly