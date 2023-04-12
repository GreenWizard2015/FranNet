import tensorflow as tf
import tensorflow.keras.layers as L
from NN.utils import sMLP

'''
Simple MLP decoder that takes condition, coords, timestep and V as input and returns corresponding V of specified 'pixels'
'''
class MLPDecoder(tf.keras.Model):
  def __init__(self, channels, blocks, residual, **kwargs):
    super().__init__(**kwargs)
    self._residual = residual
    self._channels = channels
    self._blocks = blocks(self.name)
    return
  
  @tf.function
  def _value2initstate(self, x):
    B, C = tf.shape(x)[0], tf.shape(x)[1]
    tf.assert_equal(tf.shape(x), (B, C))
    # due to unknown reasons, always false
    if C == self._channels: return x

    shp = [1, (C + self._channels) // C]
    x = tf.tile(x, shp)[..., :self._channels]
    tf.assert_equal(tf.shape(x), (B, self._channels))
    # due to unknown reasons, tf.shape(x) is (B, None), but assert_equal doesn't raise
    return tf.reshape(x, (B, self._channels))

  def call(self, condition, coords, timestep, V):
    res = self._value2initstate(V)
    initState = tf.concat([condition, coords, timestep, V], axis=-1)
    for block in self._blocks:
      state = tf.concat([initState, res], axis=-1)
      curValue = block(state)
      res = res + curValue if self._residual else curValue
      continue
    
    tf.assert_equal(tf.shape(res)[1:], (self._channels, ))
    return res
  
def _mlp_from_config(config, channels):
  def _createMlp(name):
    mlp = sMLP(config['sizes'], activation=config['activation'], name='%s/mlp' % name)
    return tf.keras.Sequential([
      mlp,
      L.Dense(channels, activation=config.get('final activation', 'linear'))
    ], name=name)
  
  if not config['shared']: return _createMlp
  # shared mlp, create it once and reuse
  shared = [None]
  def _createSharedMlp(name):
    if shared[0] is None:
      shared[0] = _createMlp(name)
      pass
    return shared[0]
  return _createSharedMlp
  
def decoder_from_config(config):
  if 'mlp' == config['name']:
    channels = config['channels']
    mlpF = _mlp_from_config(config['mlp'], channels)

    return MLPDecoder(
      channels=channels,
      blocks=lambda name: [mlpF('%s/MLP-%d' % (name, i)) for i in range(config['mlp blocks'])],
      residual=config['residual'],
    )
  
  raise ValueError(f"Unknown decoder name: {config['name']}")