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

  def _preprocessColors(self, img):
    res = self._internalRange.convert(img)
    # ensure that the image is in the RGB color space
    if 'bgr' == self._format: res = res[..., ::-1] # BGR to RGB
    return res
    
  def _src(self, img):
    tf.assert_equal(tf.shape(img)[1:], [self._imageSize, self._imageSize, 3])
    res = self._preprocessColors(img)
    if self._toGrayscale: res = tf.image.rgb_to_grayscale(res)
    return self._outputRange.convert(res)
  
  def _dest(self, cropped):
    if 'dest' in cropped: # if image provided
      res = self._preprocessColors(cropped['dest'])
      return self._outputRange.convert(res)
    
    # if samples provided
    assert 'sampled' in cropped, 'Invalid cropped: %s' % cropped
    assert 'positions' in cropped, 'Invalid cropped: %s' % cropped
    return dict(
      sampled=self._outputRange.convert(
        self._preprocessColors(cropped['sampled'])
      ),
      positions=cropped['positions']
    )

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
      'crop size': None,
      'subsample': False,
    }
    if isConfig:
      args = dict(args, **config_or_image)

    cropper = CroppingAugm.configToCropper(args, dest_size=self._imageSize)
    def _process(img):
      self._checkInput(img)
      
      cropped = cropper(img)
      src = self._src(cropped['src'])
      dest = self._dest(cropped)
      # NOTE: ALWAYS return images in the -1..1 range in RGB color space
      return(src, dest)
    
    if isConfig: return _process
    return _process(config_or_image) # apply to image directly