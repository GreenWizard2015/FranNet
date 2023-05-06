import tensorflow as tf

# diffusion (but not only) works better with squared/zero variance, but it is incorrect implementation, so I added this configurable option
class CNoiseProvider:
  def __init__(self, stddev_type):
    assert (stddev_type in ['correct', 'squared', 'zero']), 'Unknown variance type: {}'.format(stddev_type)

    self._generate_noise = lambda shape, variance: tf.random.normal(shape, stddev=tf.sqrt(variance))

    if 'squared' == stddev_type:
      self._generate_noise = lambda shape, variance: tf.random.normal(shape, stddev=variance)

    if 'zero' == stddev_type:
      self._generate_noise = lambda shape, variance: tf.zeros(shape, dtype=tf.float32)
    return
  
  def __call__(self, shape, variance):
    return self._generate_noise(shape, variance)
# end of class CNoiseProvider

def noise_provider_from_config(config):
  if isinstance(config, str):
    return CNoiseProvider(stddev_type=config)
  
  raise ValueError('Unknown noise provider: %s' % config)