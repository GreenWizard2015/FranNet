import tensorflow as tf

class CBaseModel(tf.keras.Model):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self._loss = tf.keras.metrics.Mean(name="loss")
    self._lossGrayscale = tf.keras.metrics.Mean(name="loss_gr")
    return
  
  def train_step(self, data):
    raise NotImplementedError()
  
  def test_step(self, images):
    raise NotImplementedError()
  
  def _testMetrics(self, dest, reconstructed):
    reconstructed = tf.reshape(reconstructed, tf.shape(dest)) # just in case
    # convert from -1..1 to 0..1
    dest = self.convertRange(dest, targetRange='0..1')
    reconstructed = self.convertRange(reconstructed, targetRange='0..1')

    loss = tf.losses.mse(dest, reconstructed)
    self._loss.update_state(loss)
    # calculate loss for images in a grayscale color space
    lossGrayscale = tf.losses.mse(
      tf.image.rgb_to_grayscale(dest),
      tf.image.rgb_to_grayscale(reconstructed)
    )
    self._lossGrayscale.update_state(lossGrayscale)
    
    return self.metrics_to_dict( self._loss, self._lossGrayscale )
  
  # some helper functions
  def metrics_to_dict(self, *metrics):
    return {x.name: x.result() for x in metrics}

  def convertRange(self, x, targetRange='0..1'):
    if targetRange == '-1..1': return x * 2.0 - 1.0
    
    if targetRange == '0..1': return (x + 1.0) / 2.0
    raise ValueError(f'Unknown target range "{targetRange}"')