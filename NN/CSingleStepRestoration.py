import tensorflow as tf
from NN.IRestorationProcess import IRestorationProcess

class CSingleStepRestoration(IRestorationProcess):
  def __init__(self, channels):
    super().__init__(channels)
    return
  
  def forward(self, x0):
    s = tf.shape(x0)
    T = tf.zeros(s[:-1], tf.int32)[..., None]
    return {
      "xT": tf.zeros_like(x0),
      "t": T, # dummy discrete time
      "T": tf.zeros_like(T, tf.float32), # dummy continuous time
      "x0": x0,
    }
  
  def reverse(self, value, denoiser, modelT=None, **kwargs):
    if isinstance(value, tuple):
      value = tf.zeros(value + (self._channels, ), dtype=tf.float32)

    T = [[0.0]]
    if not(modelT is None):
      T = modelT(T)[0]
    #######################
    shp = tf.shape(value)[:-1]
    T = tf.reshape(T, (1, ) * len(shp) + (-1, ))
    T = tf.tile(T, tf.concat([shp, [1]], axis=0))
    return denoiser(value, T)
  
  def calculate_loss(self, gt, predicted):
    x0 = gt["x0"]
    tf.assert_equal(tf.shape(x0), tf.shape(predicted))
    return tf.losses.mse(x0, predicted)
