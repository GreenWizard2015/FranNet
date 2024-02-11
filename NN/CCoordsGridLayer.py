import tensorflow as tf
from NN.utils import generateSquareGrid

# Input: (B, H, W, C)
# Output: (B, H, W, C + N)
# append to the input an encoded coordinates grid
# to save computation, the grid is encoded only once and then tiled to match the input shape
# this not very good for training, because decrease gradients variance
class CCoordsGridLayer(tf.keras.layers.Layer):
  def __init__(self, positionEncoder, **kwargs):
    super().__init__(**kwargs)
    self._positionEncoder = positionEncoder
    return
  
  def call(self, x):
    shp = tf.shape(x)
    B, H, W, C = shp[0], shp[1], shp[2], shp[3]
    tf.assert_equal(H, W)

    grid = generateSquareGrid(H, scale=1.0, shift=0.0)
    grid = tf.reshape(grid, (-1, 2))
    encoded = self._positionEncoder(grid)
    encC = tf.shape(encoded)[-1]
    encoded = tf.reshape(encoded, (1, H, W, encC))
    encoded = tf.tile(encoded, (B, 1, 1, 1))
    res = tf.concat([x, encoded], axis=-1)
    res = tf.ensure_shape(res, (B, H, W, C + encC))
    return res