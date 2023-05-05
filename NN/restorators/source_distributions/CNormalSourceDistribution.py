import tensorflow as tf
from .ISourceDistribution import ISourceDistribution

'''
Basic normal distribution. Just returns random values from normal distribution.
'''
class CNormalSourceDistribution(ISourceDistribution):
  def __init__(self, mean, stddev):
    self._mean = mean
    self._std = stddev
    return
  
  def sampleFor(self, x0):
    B = tf.shape(x0)[0]
    return {
      'xT': self.initialValueFor(x0),
      'T': tf.random.uniform((B, 1), 0.0, 1.0),
    }
  
  def initialValueFor(self, shape_or_values):
    shape = shape_or_values.shape if hasattr(shape_or_values, 'shape') else shape_or_values
    return tf.random.normal(shape, mean=self._mean, stddev=self._std, dtype=tf.float32)