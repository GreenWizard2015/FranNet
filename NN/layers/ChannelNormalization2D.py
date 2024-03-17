import tensorflow as tf

class ChannelNormalization2D(tf.keras.layers.Layer):
  def __init__(self, eps=1e-5, **kwargs):
    super().__init__(**kwargs)
    self.eps = eps
    return
  
  def build(self, input_shape):
    num_channels = input_shape[-1]
    self._weight = tf.Variable(
      initial_value=tf.ones((1, 1, num_channels), dtype=tf.float32),
      dtype=tf.float32,
      trainable=True,
      name='%s/weight' % self.name
    )
    self._bias = tf.Variable(
      initial_value=tf.zeros((1, 1, num_channels), dtype=tf.float32),
      dtype=tf.float32,
      trainable=True,
      name='%s/bias' % self.name
    )
    return

  def call(self, inputs):
    mean, variance = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
    normalized = (inputs - mean) / tf.sqrt(variance + self.eps)
    return self._weight * normalized + self._bias
