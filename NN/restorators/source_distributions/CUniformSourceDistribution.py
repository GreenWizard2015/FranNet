import tensorflow as tf
import tensorflow_probability as tfp
from .ISourceDistribution import ISourceDistribution

'''
Implements a uniform distribution of values in the given range.
Supported distributions: uniform, halton, sobol (random skip).
'''
class CUniformSourceDistribution(ISourceDistribution):
  def __init__(self, min, max, distribution):
    super().__init__()
    self._min = min
    self._max = max
    self._distribution = _get_distribution(distribution)
    return
  
  def sampleFor(self, x0):
    B = tf.shape(x0)[0]
    return {
      'xT': self.initialValueFor(x0),
      'T': self._distribution((B, 1)),
    }
  
  def initialValueFor(self, shape_or_values):
    noise = self._distribution(shape_or_values)
    # transform noise to the given range
    noise = self._min + (self._max - self._min) * noise
    return noise
# End of CUniformSourceDistribution

def _shuffleBatch(batch):
  indices = tf.range(tf.shape(batch)[0])
  indices = tf.random.shuffle(indices)
  return tf.gather(batch, indices)

def _get_exact_shape(tensor_or_shape):
  if isinstance(tensor_or_shape, tuple): return tensor_or_shape
  
  # hack to get correct shape, so that we can use it in sobol and halton
  shape = (tf.shape(tensor_or_shape)[0], tensor_or_shape.shape[-1])
  tf.assert_equal(tf.shape(tensor_or_shape), shape)
  return shape

def _get_distribution(name):
  name = name.lower()
  if 'uniform' == name:
    return lambda shape: tf.random.uniform(_get_exact_shape(shape), 0.0, 1.0)
  
  # TODO: find way to generate halton and sobol sequences per sample in a flattened batch, instead of shuffling them all together
  if 'halton' == name:
    def halton(shape):
      shape = _get_exact_shape(shape)
      res = tfp.mcmc.sample_halton_sequence(shape[-1], num_results=shape[0], randomized=True)
      return _shuffleBatch(res)
    return halton
  
  if 'sobol' == name:
    def sobol(shape):
      shape = _get_exact_shape(shape)
      MAX_SKIP = 100000000 # magic number
      skip = tf.random.uniform((1,), 0, MAX_SKIP, tf.int32)[0] # randomize the sequence
      res = tf.math.sobol_sample(dim=shape[-1], num_results=shape[0], skip=skip)
      return _shuffleBatch(res)
    return sobol
  
  raise ValueError('Unknown distribution')