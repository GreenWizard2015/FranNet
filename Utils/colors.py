# Contains functions for converting between color spaces.
import tensorflow as tf
import tensorflow_io as tfio
from Utils.utils import CFakeObject

def _isFloatRange(a, b):
  def f(x):
    tf.debugging.assert_greater_equal(x, tf.cast(a, x.dtype))
    tf.debugging.assert_less_equal(x, tf.cast(b, x.dtype))
    return
  return f

def _isRangeUInt8(x):
  tf.debugging.assert_integer(x)
  # x is in the 0..255 range
  tf.debugging.assert_less_equal(tf.reduce_min(x), tf.cast(0, x.dtype))
  tf.debugging.assert_greater_equal(tf.reduce_max(x), tf.cast(255, x.dtype))
  return

def _from01range(to_):
  if ('-1..1' == to_):
    return CFakeObject(
      convert=lambda x: (x * 2.0) - 1.0,
      convertBack=lambda x: (x + 1.0) / 2.0,
      check=_isFloatRange(0.0, 1.0)
    )
  
  if ('0..255' == to_):
    return CFakeObject(
      convert=lambda x: x * 255.0,
      convertBack=lambda x: tf.cast(x, tf.float32) / 255.0,
      check=_isFloatRange(0.0, 1.0)
    )
  
  raise ValueError(f'Unknown conversion: "0..1" -> "{to_}"')

def _fromUInt8Range(to_):
  if ('-1..1' == to_):
    return CFakeObject(
      convert=lambda x: (tf.cast(x, tf.float32) / 127.5) - 1.0,
      convertBack=lambda x: (x + 1.0) * 127.5,
      check=_isRangeUInt8
    )
  
  if ('0..1' == to_):
    return CFakeObject(
      convert=lambda x: tf.cast(x, tf.float32) / 255.0,
      convertBack=lambda x: x * 255.0,
      check=_isRangeUInt8
    )
  
  raise ValueError(f'Unknown conversion: "0..1" -> "{to_}"')

def convertRanges(from_, to_=None):
  # returns an object with three methods:
  # convert(x): converts x from the "from_" range to the "to_" range
  # convertBack(x): converts x from the "to_" range to the "from_" range
  # check(x): validates that x is in the "from_" range

  identity = lambda x: x # prevent tf warnings
  if to_ is None: to_ = from_
  if from_ == to_: return CFakeObject( convert=identity, convertBack=identity, check=lambda x: True )
  
  if '0..1' == from_: return _from01range(to_)
  if '0..255' == from_: return _fromUInt8Range(to_)
  raise ValueError(f'Unknown conversion: "{from_}" -> "{to_}"')

####################################
# Color space conversion functions #
####################################
# NOTE: all functions expect the input to be in the -1..1 range
#       and return the output in the -1..1 range

def _makeConverter(forwardConversion, backwardConversion, shift, scale):
  # img is in the -1..1 range
  # returns img in the -1..1 range
  shift = tf.constant(shift, dtype=tf.float32)
  scale = tf.constant(scale, dtype=tf.float32)

  def _convert(img):
    tf.assert_equal(tf.shape(img)[-1], 3)
    img = (img + 1.0) / 2.0 # convert to 0..1 range
    img = forwardConversion(img)
    # convert to -1..1 range
    res = (img - shift) / scale
    tf.assert_equal(tf.shape(res), tf.shape(img))
    return res
  
  def _convertBack(img):
    tf.assert_equal(tf.shape(img)[-1], 3)
    # convert to RGB
    img = backwardConversion( (img * scale) + shift )
    # convert to -1..1 range
    res = (img * 2.0) - 1.0
    tf.assert_equal(tf.shape(res), tf.shape(img))
    return res
  
  return CFakeObject( convert=_convert, convertBack=_convertBack, check=_isFloatRange(-1.0, 1.0) )

# LAB color space, reference implementation
def _convertRGBtoLABReference(illuminant='D65', observer='2'):
  def _convert(img):
    tf.assert_equal(tf.shape(img)[-1], 3)
    img = (img + 1.0) / 2.0 # convert to 0..1 range
    lab = tfio.experimental.color.rgb_to_lab(img, illuminant=illuminant, observer=observer)
    # lab is in [0..100, -128..127, -128..127] range
    # convert to [-50..50, -128..127, -128..127] range
    lab = lab - tf.constant([50.0, 0.0, 0.0], dtype=lab.dtype)
    # convert to -1..1 range
    res = lab / tf.constant([50.0, 127.5, 127.5], dtype=lab.dtype)
    tf.assert_equal(tf.shape(res), tf.shape(img))
    return res
  
  def _convertBack(img):
    tf.assert_equal(tf.shape(img)[-1], 3)
    # img is in the -1..1 range, convert to [0..100, -128..127, -128..127] range
    img = img * tf.constant([50.0, 127.5, 127.5], dtype=img.dtype)
    # convert to [0..100, 0..255, 0..255] range
    lab = img + tf.constant([50.0, 0.0, 0.0], dtype=img.dtype)
    # convert to RGB
    rgb = tfio.experimental.color.lab_to_rgb(lab, illuminant=illuminant, observer=observer)
    # convert to -1..1 range
    res = (rgb * 2.0) - 1.0
    tf.assert_equal(tf.shape(res), tf.shape(img))
    return res
  
  return CFakeObject(
    convert=_convert,
    convertBack=_convertBack,
    check=_isFloatRange(-1.0, 1.0)
  )

# LAB color space
def convertRGBtoLAB(illuminant='D65', observer='2'):
  return _makeConverter(
    forwardConversion=lambda x: tfio.experimental.color.rgb_to_lab(x, illuminant=illuminant, observer=observer),
    backwardConversion=lambda x: tfio.experimental.color.lab_to_rgb(x, illuminant=illuminant, observer=observer),
    # range is [0..100, -128..127, -128..127]
    shift=[50.0, 0.0, 0.0],
    scale=[50.0, 127.5, 127.5]
  )

# HSV color space
def convertRGBtoHSV():
  return _makeConverter(
    forwardConversion=tf.image.rgb_to_hsv,
    backwardConversion=tf.image.hsv_to_rgb,
    # range is [0..360, 0..1, 0..1]
    shift=[180.0, 0.5, 0.5],
    scale=[180.0, 0.5, 0.5]
  )

# HSL color space
def convertRGBtoHSL():
  return _makeConverter(
    forwardConversion=tf.image.rgb_to_hsv,
    backwardConversion=tf.image.hsv_to_rgb,
    # range is [0..360, 0..1, 0..1]
    shift=[180.0, 0.5, 0.5],
    scale=[180.0, 0.5, 0.5]
  )

# YUV color space
def convertRGBtoYUV():
  return _makeConverter(
    forwardConversion=tf.image.rgb_to_yuv,
    backwardConversion=tf.image.yuv_to_rgb,
    # range is [0..1, -0.436..0.436, -0.615..0.615]
    shift=[0.0, -0.436, -0.615],
    scale=[1.0, 0.436 * 2.0, 0.615 * 2.0]
  )

# YIQ color space
def convertRGBtoYIQ():
  return _makeConverter(
    forwardConversion=tf.image.rgb_to_yiq,
    backwardConversion=tf.image.yiq_to_rgb,
    # range is [0..1, -0.596..0.596, -0.523..0.523]
    shift=[0.0, -0.596, -0.523],
    scale=[1.0, 0.596 * 2.0, 0.523 * 2.0]
  )

# BGR color space
def convertRGBtoBGR():
  reverse = lambda x: x[..., ::-1]
  return CFakeObject(
    convert=reverse,
    convertBack=reverse,
    check=_isFloatRange(-1.0, 1.0)
  )

# identity/RGB color space
def convertRGBtoRGB():
  return CFakeObject(
    convert=lambda x: x,
    convertBack=lambda x: x,
    check=_isFloatRange(-1.0, 1.0)
  )

RGB_CONVERSIONS = {
  'lab': convertRGBtoLAB,
  'hsv': convertRGBtoHSV,
  'hsl': convertRGBtoHSL,
  'yuv': convertRGBtoYUV,
  'yiq': convertRGBtoYIQ,
  'bgr': convertRGBtoBGR,
  'rgb': convertRGBtoRGB,
  'identity': convertRGBtoRGB,
}
###################
# some dumb tests #
###################
if __name__ == '__main__':
  import numpy as np
  
  def _testConversion(converter):
    values = np.linspace(-1.0, 1.0, num=16 * 1024 * 3)
    values = values.reshape((16, 1024, 3))
    src = tf.constant(values, dtype=tf.float32)
    dest = converter.convert(src)
    newImg = converter.convertBack( dest )
    tf.assert_equal(tf.shape(newImg), tf.shape(src))
    tf.debugging.assert_near(src, newImg, atol=1e-3)
    # check that the values are in the expected range
    tf.debugging.assert_greater_equal(tf.reduce_min(dest), -1.0)
    tf.debugging.assert_less_equal(tf.reduce_max(dest), 1.0)
    return
  
  def testReference():
    actualConverter = convertRGBtoLAB()
    referenceConverter = _convertRGBtoLABReference()
    values = np.linspace(-1.0, 1.0, num=16 * 1024 * 3, dtype=np.float32)
    values = values.reshape((16, 1024, 3))

    A = actualConverter.convert(values)
    B = referenceConverter.convert(values)
    tf.debugging.assert_near(A, B, atol=1e-5)

    AB = actualConverter.convertBack(B)
    BB = referenceConverter.convertBack(A)
    tf.debugging.assert_near(AB, BB, atol=1e-5)
    return
  
  testReference()

  for name, converter in RGB_CONVERSIONS.items():
    print(f'Checking {name}... ', end='')
    _testConversion(converter())
    print('OK')
    continue
  print('OK')
  pass
