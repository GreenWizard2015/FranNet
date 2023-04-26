import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import tensorflow.keras.layers as L
from NN.CCoordsEncodingLayer import CCoordsEncodingLayer

def normVec(x):
  V, L = tf.linalg.normalize(x, axis=-1)
  V = tf.where(tf.math.is_nan(V), 0.0, V)
  return(V, L)

def ensure4d(src):
  return tf.reshape(src, tf.concat([tf.shape(src)[:3], (-1,)], axis=-1)) # 3d/4d => 4d

def extractInterpolated(data, pos):
  s = tf.shape(data)
  hw = tf.cast(tf.stack([s[1], s[2]]), pos.dtype)
  coords = pos * (hw - 1.0)
  return tfa.image.interpolate_bilinear(data, coords, indexing='xy')

class CFlatCoordsEncodingLayer(tf.keras.layers.Layer):
  def __init__(self, N=32, **kwargs):
    super().__init__(**kwargs)
    self._enc = CCoordsEncodingLayer(N)
    return

  def call(self, x):
    B = tf.shape(x)[0]
    tf.assert_equal(tf.shape(x)[:-1], (B, ))
    x = tf.cast(x, tf.float32)[..., None]
    return self._enc(x)[:, 0]
#######################################
class sMLP(tf.keras.layers.Layer):
  def __init__(self, sizes, activation='linear', dropout=0.05, **kwargs):
    super().__init__(**kwargs)
    layers = []
    for i, sz in enumerate(sizes):
      if 0.0 < dropout:
        layers.append(L.Dropout(dropout, name='%s/dropout-%i' % (self.name, i)))
      layers.append(L.Dense(sz, activation=activation, name='%s/dense-%i' % (self.name, i)))
      continue
    self._F = tf.keras.Sequential(layers, name=self.name + '/F')
    return
  
  def call(self, x, **kwargs):
    return self._F(x, **kwargs)
