import tensorflow as tf
import tensorflow.keras.activations as activations

# Apply sequence of extractors to the same input and combine them by using learned gate skip connections
class CCombinedExtractor(tf.keras.Model):
  def __init__(self, extractors, weightsActivation, **kwargs):
    super().__init__(**kwargs)
    extractors = [f('%s/Extractor-%d' % (self.name, i)) for i, f in enumerate(extractors)]
    self._extractors = extractors[1:]
    self._rootExtractor = extractors[0]
    self._weights = tf.Variable(
      initial_value=tf.zeros((len(self._extractors),), dtype=tf.float32),
      trainable=True, name='weights'
    )
    self._weightsActivation = weightsActivation
    return
  
  def call(self, features, pos, training=None):
    res = self._rootExtractor(features, pos, training=training)
    shape = tf.shape(res)
    for i, extractor in enumerate(self._extractors):
      x = extractor(features, pos, training=training)
      tf.assert_equal(tf.shape(x), shape)
      res += x * self._weightsActivation(self._weights[i])
      continue
    tf.assert_equal(tf.shape(res), shape)
    return res
# End of CCombinedExtractor

def combined_extractor_from_config(config, extractors):
  weightsActivation = config['weights activation']
  if 'squared tanh' == weightsActivation:
    # always positive, in [0, 1]
    weightsActivation = lambda x: tf.square( tf.nn.tanh(x) )
  else:
    weightsActivation = activations.get(weightsActivation)

  return lambda name: CCombinedExtractor(
    extractors,
    weightsActivation=weightsActivation,
    name=name
  )