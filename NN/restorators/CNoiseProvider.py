import tensorflow as tf

class CNoiseProvider:
  def __init__(self, stddev_type):
    mapping = {
      'correct': self._generate_normal_noise,
      'normal': self._generate_normal_noise,
      'squared': self._generate_squared_noise,
      'zero': self._generate_zero_noise
    }
    self._generate_noise = mapping.get(stddev_type, None)
    assert self._generate_noise is not None, f'Unknown stddev type: {stddev_type}'
    return
  
  def __call__(self, **kwargs):
    # ensure that shape and sigma are provided as named arguments
    shape = kwargs['shape']
    sigma = kwargs['sigma']
    return self._generate_noise(shape, sigma)
  
  def _generate_normal_noise(self, shape, sigma):
    return tf.random.normal(shape) * sigma
  
  def _generate_squared_noise(self, shape, sigma):
    return tf.random.normal(shape) * tf.square(sigma)
  
  def _generate_zero_noise(self, shape, sigma):
    return tf.zeros(shape, dtype=tf.float32)
# end of class CNoiseProvider

def noise_provider_from_config(config):
  if isinstance(config, str):
    return CNoiseProvider(stddev_type=config)
  
  raise ValueError('Unknown noise provider: %s' % config)