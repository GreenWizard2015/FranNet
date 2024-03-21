import tensorflow as tf
from .CCoordsEncodingLayer import CCoordsEncodingLayerV1
from .CCoordsEncodingLayerV2 import CCoordsEncodingLayerV2 as CCoordsEncodingLayer
from .CCoordsGridLayer import CCoordsGridLayer
from .CFixedSinCosEncoding import CFixedSinCosEncoding

# Old and incorrect implementation of the encoding layer
class CFlatCoordsEncodingLayer_OLD(tf.keras.layers.Layer):
  def __init__(self, N=32, **kwargs):
    super().__init__(**kwargs)
    self._enc = CCoordsEncodingLayerV1(N)
    return

  def call(self, x):
    B = tf.shape(x)[0]
    tf.assert_equal(tf.shape(x)[:-1], (B, ))
    x = tf.cast(x, tf.float32)[..., None]
    return self._enc(x)[:, 0]
  
# Correct implementation of the encoding layer
class CFlatCoordsEncodingLayer(tf.keras.layers.Layer):
  def __init__(self, encoder, **kwargs):
    super().__init__(**kwargs)
    self._encoder = encoder
    return

  def call(self, x):
    B = tf.shape(x)[0]
    tf.assert_rank(x, 2, "Input should be a 2D tensor")
    tf.assert_equal(tf.shape(x)[:-1], (B, ))
    x = tf.cast(x, tf.float32)[:, None, :]
    return self._encoder(x)[:, 0]
  
def encoding_from_config(config):
  if isinstance(config, str):
    config = { 'name': config, 'N': 32 }

  if isinstance(config, dict):
    name = config['name']
    params = { k: v for k, v in config.items() if k != 'name' }
    if 'learned' == name: return CFlatCoordsEncodingLayer_OLD(**params)
    if 'fixed' == name: return CFixedSinCosEncoding(**params)

    if 'learned v2' == name: return CFlatCoordsEncodingLayer(
      encoder=CCoordsEncodingLayerV1(**params)
    )
    if 'learned v3' == name: return CFlatCoordsEncodingLayer(
      encoder=CCoordsEncodingLayer(**params)
    )    
    raise ValueError(f"Unknown encoding name: {name}")

  raise ValueError(f"Unknown encoding config: {config}")
