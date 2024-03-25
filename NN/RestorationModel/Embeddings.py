'''
Contains the code for the Embeddings class, which is used to create the embeddings for the model.
it takes in an integer value and returns the embeddings for the model (1, ?)
'''

import tensorflow as tf

class CNumberEmbeddings(tf.keras.layers.Layer):
  def __init__(self, N, D, **kwargs):
    super().__init__(**kwargs)
    self._N = N
    self._embeddings = tf.keras.layers.Embedding(
      input_dim=N,
      output_dim=D,
      embeddings_initializer='glorot_uniform',
      mask_zero=False,
    )
    return
  
  def call(self, x):
    tf.assert_less(x, self._N)
    tf.debugging.assert_greater_equal(x, 0)
    tf.assert_rank(x, 0)
    x = tf.reshape(x, (1, 1))
    return self._embeddings(x)[:, 0]
# End of CNumberEmbeddings
  
class CEncodedEmbeddings(tf.keras.layers.Layer):
  def __init__(self, encoding, N, **kwargs):
    super().__init__(**kwargs)
    self._N = N
    self._encoding = encoding
    return
  
  def call(self, x):
    tf.assert_less(x, self._N)
    tf.debugging.assert_greater_equal(x, 0)
    tf.assert_rank(x, 0)
    x = tf.reshape(x, (1, 1))
    x = tf.cast(x, dtype=tf.float32) / self._N
    return self._encoding(x)