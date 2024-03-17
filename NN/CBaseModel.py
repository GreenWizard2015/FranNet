import tensorflow as tf
from Utils.utils import CFakeObject
import Utils.colors as colors
from .CKIDMetric import CKIDMetric

def _makeConverter(format):
  format = format.lower()
  identity = lambda x: x # prevent tf warnings
  # Returns an object with two methods:
  # convert(x): converts x from the "RGB, -1..1" range to the target format
  # convertBack(x): converts x from the target format to the "RGB, -1..1" range
  if isinstance(format, str):
    if ('identity' == format) or ('rgb' == format): # do nothing
      return CFakeObject( convert=identity, convertBack=identity, check=lambda x: True )
    
    if format in colors.RGB_CONVERSIONS:
      return colors.RGB_CONVERSIONS[format]()
    pass
  # if dict with 'name' = 'lab' is passed
  if isinstance(format, dict) and ('name' in format):
    name = format['name'].lower()
    args = {**format}
    args.pop('name')

    if name in colors.RGB_CONVERSIONS:
      return colors.RGB_CONVERSIONS[name](**args)
    pass

  raise ValueError(f'Unknown format "{format}"')

class CBaseModel(tf.keras.Model):
  def __init__(self, format='identity', **kwargs):
    super().__init__(**kwargs)
    self._loss = tf.keras.metrics.Mean(name="loss")
    self._lossGrayscale = tf.keras.metrics.Mean(name="loss_gr")
    self._converter = _makeConverter(format)
    self._kidMetric = CKIDMetric(name="kid")
    return

  def train_step(self, data):
    raise NotImplementedError()
  
  def test_step(self, images):
    raise NotImplementedError()
  
  def _testMetrics(self, dest, reconstructed):
    reconstructed = tf.reshape(reconstructed, tf.shape(dest)) # just in case
    # expecting dest and reconstructed to be in the -1..1 range, in RGB color space
    # convert them to 0..1
    dest = (dest + 1.0) / 2.0
    # reconstructed should already be in 0..1 and in RGB color space!!!
    # reconstructed = self._converter.convertBack(reconstructed)
    reconstructed = (reconstructed + 1.0) / 2.0 # convert to 0..1
    
    loss = tf.losses.mse(dest, reconstructed)
    self._loss.update_state(loss)
    # calculate KID metric on the images, 0..1 range
    self._kidMetric.update_state(dest, reconstructed)
    # calculate loss for images in a grayscale color space
    lossGrayscale = tf.losses.mse(
      tf.image.rgb_to_grayscale(dest),
      tf.image.rgb_to_grayscale(reconstructed)
    )
    self._lossGrayscale.update_state(lossGrayscale)
    
    return self.metrics_to_dict( self._loss, self._lossGrayscale, self._kidMetric )
  
  # some helper functions
  def metrics_to_dict(self, *metrics):
    return {x.name: x.result() for x in metrics}