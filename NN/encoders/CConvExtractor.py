import tensorflow as tf
import tensorflow.keras.layers as L
import numpy as np
import NN.utils as NNU

class CBasicConvMixer(tf.keras.Model):
  def __init__(self, latentDim, **kwargs):
    super().__init__(**kwargs)
    self._latentDim = latentDim
    self._model = tf.keras.Sequential([
      L.SeparableConv2D(latentDim, 1, padding='same', activation='relu'),
      L.SeparableConv2D(latentDim, 2, padding='valid', activation='relu'),
      L.SeparableConv2D(latentDim, 2, padding='valid', activation='relu'),
      L.Flatten(),
      L.Dense(latentDim, activation='relu'),
    ], name=self.name + '/model')
    return
  
  def call(self, inputs, training=None):
    return self._model(inputs, training=training)
  
  @property
  def input_shape(self):
    return (None, 3, 3, None) # hack to make it work with CConvExtractor without building the model
# End of CBasicConvMixer

class CConvExtractor(tf.keras.Model):
  def __init__(self, localMixer, padding=None, **kwargs):
    super().__init__(**kwargs)
    self._localMixer = localMixer(self.name + '/LocalMixer')
    inpShape = self._localMixer.input_shape
    assert len(inpShape) == 4, 'local mixer must accept 2D input'
    assert inpShape[1] == inpShape[2], 'local mixer must accept square input'
    self._patchSize = inpShape[1]
    self._padding = padding
    return
  
  def _patchMesh(self):
    d = self._patchSize // 2
    indices = tf.range(-d, d + 1)
    indices = tf.meshgrid(indices, indices)
    indices = tf.stack(indices, axis=-1)
    indices = tf.reshape(indices, (self._patchSize * self._patchSize, 2))
    return indices
  
  def _patchIndices(self, H, W):
    tf.assert_equal(H, W) # feature map must be square
    indices = self._patchMesh()
    indices = indices[:, 0] + indices[:, 1] * W
    tf.assert_equal(tf.size(indices), self._patchSize * self._patchSize)
    tf.assert_rank(indices, 1) # indices must be 1D tensor
    return indices
  
  def _coordGrid(self, size):
    coordsGrid = 0.0 + tf.cast(self._patchMesh(), tf.float32) # shift to center of pixel
    coordsGrid = coordsGrid / tf.cast(size, tf.float32) # normalize
    coordsGrid = tf.reshape(coordsGrid, (1, self._patchSize, self._patchSize, 2))
    return coordsGrid

  def _pixelInfo(self, pos, size):
    size = tf.cast(size, tf.float32)
    pos = tf.cast(pos, tf.float32)
    apos = pos * (size - 1)
    tf.assert_equal(tf.shape(apos), tf.shape(pos))
    apos = tf.floor(apos)
    res = (apos + 0.5) / size
    tf.assert_equal(tf.shape(res), tf.shape(pos))
    return {
      'center normalized': res, # (0 0) -> (0.5 0.5), (1 1) -> (size - 0.5, size - 0.5)
      'center indices': tf.cast(apos, tf.int32), # (0 0) -> (0 0), (1 1) -> (size - 1, size - 1)
      'pos': pos,
    }
  
  def _extractPatchesRaw(self, data, centerPos):
    B = tf.shape(centerPos)[0]
    N = tf.shape(centerPos)[1]
    tf.assert_equal(tf.shape(centerPos), (B, N, 2))
    tf.assert_equal(centerPos.dtype, tf.int32)

    C = data.shape[-1]
    P = self._patchSize * self._patchSize
    padding = self._patchSize // 2
    # prepare data for patch extraction, pad with zeros and flatten inner dimensions
    data = tf.pad(
      data, 
      [[0, 0], [padding, padding], [padding, padding], [0, 0]],
      **self._padding
    )
    s = tf.shape(data)[1:3]
    data = tf.reshape(data, (B, -1, C))

    # find flat indices of patches
    flatIndicesMask = self._patchIndices(s[0], s[1])
    centerPos = padding + centerPos
    flatIndices = centerPos[:, :, 0] + centerPos[:, :, 1] * s[0]
    tf.assert_equal(tf.shape(flatIndices), (B, N))
    indices = flatIndices[:, :, None] + flatIndicesMask[None, None, :]
    tf.assert_equal(tf.shape(indices), (B, N, P))

    indices = tf.reshape(indices, (B, N * P))
    # extract patches
    patches = tf.gather(data, indices, batch_dims=1)
    tf.assert_equal(tf.shape(patches), (B, N * P, C))
    patches = tf.reshape(patches, (B * N, self._patchSize, self._patchSize, C))
    return patches

  def _pixelMetrics(self, pos, grid):
    s = tf.shape(pos['pos'])
    BN = s[0] * s[1]
    rounded_pos_center = tf.reshape(pos['center normalized'], (BN, 1, 1, 2))
    coordsGrid = grid + rounded_pos_center
    tf.assert_equal(tf.shape(coordsGrid), (BN, self._patchSize, self._patchSize, 2))
    # find distances to each pixel in the patch
    diff = coordsGrid - tf.reshape(pos['pos'], (BN, 1, 1, 2))
    tf.assert_equal(tf.shape(diff), (BN, self._patchSize, self._patchSize, 2))
    return NNU.normVec(diff)
    
  def _extractPatches(self, data, pos):
    tf.assert_rank(data, 4)
    data_size = tf.shape(data)[1:3]
    B = tf.shape(pos)[0]
    N = tf.shape(pos)[1]
    C = data.shape[-1]
    
    centerPos = self._pixelInfo(pos, data_size)
    patches = self._extractPatchesRaw(data, centerPos['center indices'])
    tf.assert_equal(tf.shape(patches), (B * N, self._patchSize, self._patchSize, C))
    
    metrics = self._pixelMetrics(centerPos, grid=self._coordGrid(data_size))
    res = tf.concat([patches, *metrics], axis=-1)
    # add on top of each patch its distance to the corresponding position
    tf.assert_equal(tf.shape(res)[:-1], (B * N, self._patchSize, self._patchSize))
    return res

  def call(self, features, pos, training=None):
    B = tf.shape(pos)[0]
    N = tf.shape(pos)[1]
    # extract latent vectors from each 2D feature map
    latent = [self._extractPatches(data, pos) for data in features]
    
    # concatenate all latent vectors
    latent = tf.concat(latent, axis=-1)
    tf.assert_rank(latent, 4)
    tf.assert_equal(tf.shape(latent)[:3], (B * N, self._patchSize, self._patchSize))
    # mix latent vectors
    res = self._localMixer(latent, training=training)
    tf.assert_equal(tf.shape(res)[:1], (B * N,))
    tf.assert_rank(res, 2)
    return res
# End of CConvExtractor

def conv_extractor_from_config(config, latentDim):
  padding = config.get('padding', dict(mode='CONSTANT', constant_values=0))
  return lambda name: CConvExtractor(
    localMixer=lambda nm: CBasicConvMixer(latentDim=latentDim, name=nm),
    padding=padding,
    name=name
  )