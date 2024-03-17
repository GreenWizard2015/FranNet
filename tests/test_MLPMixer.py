import tensorflow as tf
import numpy as np
import pytest
from NN.layers import MixerConvLayer

def test_same_output_shape():
  mixer = MixerConvLayer(token_mixing=2048, channel_mixing=3)
  mixer.build(input_shape=(None, 64, 64, 3))
  assert np.array_equal(mixer.compute_output_shape((None, 64, 64, 3)), [None, 64, 64, 3]), "Output shape is not as expected"
  return

def test_same_output_shape_2():
  mixer = MixerConvLayer(token_mixing=3, channel_mixing=3)
  x = tf.random.normal((1, 64, 64, 3))
  y = mixer(x)
  assert np.array_equal(y.shape, [1, 64, 64, 3]), "Output shape is not as expected"
  return