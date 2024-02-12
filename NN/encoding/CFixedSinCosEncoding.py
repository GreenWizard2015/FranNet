import tensorflow as tf
import numpy as np

class CFixedSinCosEncoding(tf.keras.layers.Layer):
  def __init__(self, N=32, **kwargs):
    super().__init__(**kwargs)
    self._N = np.ceil(N / 2).astype(np.int32)
    N = self._N * 2
    inv_freq = np.power(10000, -np.arange(0, N, 2) / np.float32(N))
    self._freq = tf.constant(inv_freq, dtype=tf.float32)[None, None]
    return
  
  def call(self, x):
    shape = tf.shape(x)
    B = shape[0]
    C = shape[-1]
    x = tf.reshape(x, (B, -1, 1))
    M = tf.shape(x)[1]
    
    # multiply x by frequency
    x = tf.matmul(x, self._freq)
    tf.assert_equal(tf.shape(x), (B, M, self._N))
    x = tf.concat([tf.sin(x), tf.cos(x)], axis=-1)
    tf.assert_equal(tf.shape(x), (B, M, 2 * self._N))

    newShape = tf.concat([shape[:-1], [C * self._N * 2]], axis=0)
    return tf.reshape(x, newShape)

if '__main__' == __name__: # test
  x = tf.linspace(0., 1., 1002)
  for shp in [(1, -1), (1, -1, 1), (1, -1, 2), (1, -1, 3)]:
    xhat = tf.reshape(x, shp)
    emb = CFixedSinCosEncoding()(xhat)
    print(xhat.shape, emb.shape)
    continue
  print('Done.')