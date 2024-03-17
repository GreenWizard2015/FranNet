import tensorflow as tf
from NN.layers import ChannelNormalization2D

class MixerConvLayer(tf.keras.layers.Layer):
  def __init__(self, token_mixing=None, channel_mixing=None, **kwargs):
    super().__init__(**kwargs)
    if token_mixing is None:
      token_mixing = channel_mixing
    if channel_mixing is None:
      channel_mixing = token_mixing

    assert token_mixing is not None, "Either token_mixing or channel_mixing must be specified"
    assert channel_mixing is not None, "Either token_mixing or channel_mixing must be specified"    
    self._tokenMixing = token_mixing
    self._channelMixing = channel_mixing
    return
  
  def build(self, input_shape):
      H, W, num_channels = input_shape[-3], input_shape[-2], input_shape[-1]
      self._channels = num_channels
      token_mixing = self._tokenMixing
      channel_mixing = self._channelMixing

      self.c_norm1 = ChannelNormalization2D(name='%s/ChannelNorm1' % self.name)
      self.convTMix = tf.keras.Sequential([ # by H
        tf.keras.layers.Conv1D(token_mixing, 1, name='%s/TMix/Conv1D-1' % self.name, activation='gelu'),
        tf.keras.layers.Conv1D(num_channels * H, 1, name='%s/TMix/Conv1D-final' % self.name)
      ], name='%s/TokenMixing' % self.name)

      self.c_norm2 = ChannelNormalization2D(name='%s/ChannelNorm2' % self.name)
      self.convCMix = tf.keras.Sequential([ # by W
        tf.keras.layers.Conv1D(channel_mixing, 1, name='%s/CMix/Conv1D-1' % self.name, activation='gelu'),
        tf.keras.layers.Conv1D(num_channels * W, 1, name='%s/CMix/Conv1D' % self.name)
      ], name='%s/ChannelMixing' % self.name)

  def call(self, inputs):
    shp = tf.shape(inputs)
    B, H, W, C = [shp[i] for i in range(4)]
    tf.assert_equal(C, self._channels)
    C = self._channels

    x = tf.transpose(inputs, [0, 2, 1, 3]) # [B, W, H, C] -> [B, H, W, C]
    x = tf.reshape(x, [B, W, H * C]) # [B, H, W, C] -> [B, W, H * C]
    centred1 = self.c_norm1(x) # Normalization of H * C
    mix1 = self.convTMix(centred1) # Mixing of H * C
    mix1 = tf.reshape(mix1, [B, W, H, C]) # [B, W, H * C] -> [B, W, H, C]
    mix1 = tf.transpose(mix1, [0, 2, 1, 3]) # [B, W, H, C] -> [B, H, W, C]
    skip1 = inputs + mix1 # Skip connection

    x = tf.reshape(skip1, [B, H, W * C]) # [B, H, W, C] -> [B, H, W * C]
    centred2 = self.c_norm2(x) # Normalization of W * C
    conv_mix2 = self.convCMix(centred2) # Mixing of W * C
    conv_mix2 = tf.reshape(conv_mix2, shp) # [B, H, W, C]

    res = tf.ensure_shape(conv_mix2 + skip1, [None, None, None, C]) # Skip connection
    return res