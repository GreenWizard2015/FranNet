import tensorflow as tf
import numpy as np
from NN.layers import Patches
import pytest

@pytest.mark.parametrize('patch_size', [4, 8, 16])
def test_output_shape(patch_size):
  patches = Patches(patch_size=patch_size)
  x = tf.random.normal((1, 64, 64, 3))
  y = patches(x).numpy()
  assert np.allclose(y.shape, [1, 64 // patch_size * 64 // patch_size, 3 * patch_size * patch_size]), "Output shape is not as expected"
  return

# test raise error if patch size is not compatible
def test_raise_error():
  patches = Patches(patch_size=5)
  x = tf.random.normal((1, 64, 64, 3))
  with pytest.raises(AssertionError):
    y = patches(x)
  return