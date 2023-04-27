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

if __name__ == '__main__': # very dumb tests
  PV = -123.456
  extractor = CConvExtractor(
    localMixer=lambda nm: CBasicConvMixer(latentDim=1, name=nm),
    padding=dict(mode='CONSTANT', constant_values=PV)
  )
  # test data. From 0 to N
  B = 3
  S = 4
  C = 1
  data = tf.reshape(tf.range(B * S * S * C, dtype=tf.float32), (B, S, S, C))

  def test_patchMesh():
    res = extractor._patchMesh().numpy()
    assert np.allclose(res, [
      [-1, -1], [0, -1], [1, -1],
      [-1, 0], [0, 0], [1, 0],
      [-1, 1], [0, 1], [1, 1]
    ])
    return
  
  def test_coordGrid():
    expected = np.array([
      [[-1.0, -1.0], [0.0, -1.0], [1.0, -1.0]],
      [[-1.0,  0.0], [0.0,  0.0], [1.0,  0.0]],
      [[-1.0,  1.0], [0.0,  1.0], [1.0,  1.0]],
    ])
    
    for N in range(1, 16):
      grid = extractor._coordGrid(N).numpy()
      assert grid.shape == (1, 3, 3, 2)
      assert np.allclose(grid, expected / N)
      continue
    return
  
  def test_patchIndices():
    indices = extractor._patchIndices(7, 7).numpy()
    assert indices.shape == (3 * 3, )
    assert np.allclose(indices, [
      -8, -7, -6,
      -1,  0,  1,
       6,  7,  8,
    ])
    return
  
  def test_pixelInfo():
    pos = tf.constant([
      [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]],
    ])
    size = tf.constant([3, 3], dtype=tf.int32)
    halfPixel = 0.5 / 3
    res = extractor._pixelInfo(pos, size)
    res = {k: v.numpy() for k, v in res.items()}
    assert np.allclose(res['center normalized'][0, 0], [ (0 + 1) * halfPixel, (0 + 1) * halfPixel])
    assert np.allclose(res['center normalized'][0, 1], [ (2 + 1) * halfPixel, (2 + 1) * halfPixel])
    assert np.allclose(res['center normalized'][0, 2], [ (4 + 1) * halfPixel, (4 + 1) * halfPixel])
    
    assert np.allclose(res['center indices'], [
      [[0, 0], [1, 1], [2, 2]],
    ])
    assert np.allclose(res['pos'], [
      [[0, 0], [0.5, 0.5], [1, 1]],
    ])
    return
  
  def test_extractPatchesRaw():
    pos = tf.constant([
      [[0, 0], [1, 1], [2, 2], [3, 3]],
      [[0, 0], [1, 0], [2, 2], [3, 3]],
      [[0, 0], [0, 1], [2, 2], [3, 3]],
    ])
    N = 4
    res = extractor._extractPatchesRaw(data, pos).numpy()
  
    assert res.shape == (B * N, 3, 3, 1)
    for i in range(B):
      # corner patch 0,0
      x = res[i * N, ..., 0]
      assert np.allclose(x[0], [PV, PV, PV])
      assert np.allclose(x[:, 0], [PV, PV, PV])
      expected = data[i, :2, :2, 0]
      assert np.allclose(x[1:,  1:], expected)
  
      # inner patch 2,2
      x = res[i * N + 2, ..., 0]
      expected = data[i, 1:4, 1:4, 0]
      assert np.allclose(x, expected)

      # corner patch 3,3
      x = res[i * N + 3, ..., 0]
      assert np.allclose(x[2], [PV, PV, PV])
      assert np.allclose(x[:, 2], [PV, PV, PV])
      expected = data[i, -2:, -2:, 0]
      assert np.allclose(x[:2, :2], expected)
      continue

    # bottom right corner with 0 padding
    x = res[3, ..., 0]
    assert np.allclose(x[0], [10, 11, PV])
    assert np.allclose(x[1], [14, 15, PV])
    assert np.allclose(x[2], [PV, PV, PV])

    # 1,0 at batch 1, position 1
    x = res[1 * N + 1, ..., 0]
    assert np.allclose(x[0], [PV, PV, PV])
    x = x - (S * S - 1)
    assert np.allclose(x[1], [ 1,  2,  3])
    assert np.allclose(x[2], [ 5,  6,  7])

    # 0, 1 at batch 2, position 1
    x = res[2 * N + 1, ..., 0]
    assert np.allclose(x[:, 0], [PV, PV, PV]) # left column is padding
    expected = data[2, 0:3, :2, 0].numpy()
    assert np.allclose(x[:, 1:], expected) 
    return
  
  def test_pixelMetrics_center():
    grid = extractor._coordGrid(32) # 32x32 grid
    onePixel = 1.0 / 32.0
    pos = tf.constant([[ [0.5 * onePixel, 0.5 * onePixel], ]])
    pos = extractor._pixelInfo(pos, 32)
    metrics = extractor._pixelMetrics(pos, grid)
    vec, L = [m.numpy() for m in metrics]
    # center pixel should have 0 vector and 0 length
    assert np.allclose(L[0, 1, 1], [0])
    assert np.allclose(vec[0, 1, 1], [0, 0])
    # (0, 0), (0, 2), (2, 0), (2, 2) should have same length, because they are on the same diagonal
    assert np.allclose(L[0, 0, 0], L[0, 0, 2])
    assert np.allclose(L[0, 0, 0], L[0, 2, 0])
    assert np.allclose(L[0, 0, 0], L[0, 2, 2])
    # (0, 0), (0, 2), (2, 0), (2, 2) should have same abs vector, because they are on the same diagonal
    assert np.allclose(np.abs(vec[0, 0, 0]), np.abs(vec[0, 0, 2]))
    assert np.allclose(np.abs(vec[0, 0, 0]), np.abs(vec[0, 2, 0]))
    assert np.allclose(np.abs(vec[0, 0, 0]), np.abs(vec[0, 2, 2]))

    # (0, 1), (1, 0), (1, 2), (2, 1) should have same length, because they are on the same diagonal
    assert np.allclose(L[0, 0, 1], L[0, 1, 0])
    assert np.allclose(L[0, 0, 1], L[0, 1, 2])
    assert np.allclose(L[0, 0, 1], L[0, 2, 1])

    # max length should be equal to sqrt(onePixel^2 + onePixel^2)
    assert np.allclose(L.max(), np.sqrt(2) * onePixel)
    return
  
  def test_pixelMetrics_zero():
    grid = extractor._coordGrid(32) # 32x32 grid
    pos = tf.constant([[ [0.0, 0.0], ]])
    pos = extractor._pixelInfo(pos, 32)
    metrics = extractor._pixelMetrics(pos, grid)
    vec, L = [m.numpy() for m in metrics]
    # L in first quadrant should be same for all pixels
    quadrant = L[0, :2, :2]
    assert np.allclose(quadrant, quadrant[0, 0])
    # L at row 2 should be same as in column 2
    assert np.allclose(L[0, 2, :, 0], L[0, :, 2, 0])
    return
  
  def test_extractPatches():
    # corners
    pos = tf.constant([
      [[0, 0], [0, 1], [1, 0], [1, 1]]
    ] * B, dtype=tf.float32)
    N = 4
    patches = extractor._extractPatches(data, pos).numpy()
    tf.assert_equal(patches.shape, [B * N, 3, 3, 1 + 3])
    for i in range(B):
      # first patch is at (0, 0)
      x = patches[i * N + 0, 1:, 1:, 0]
      expected = data[i, :2, :2, 0].numpy()
      assert np.allclose(x, expected)
      # second patch is at (0, 1)
      x = patches[i * N + 1, :2, 1:, 0]
      expected = data[i, -2:, :2, 0].numpy()
      assert np.allclose(x, expected)
      # third patch is at (1, 0)
      x = patches[i * N + 2, 1:, :2, 0]
      expected = data[i, :2, -2:, 0].numpy()
      assert np.allclose(x, expected)
      # fourth patch is at (1, 1)
      x = patches[i * N + 3, :2, :2, 0]
      expected = data[i, -2:, -2:, 0].numpy()
      assert np.allclose(x, expected)
      continue
    return
  
  def test_extractPatches_independent():
    # corners
    pos = tf.constant([
      [[0, 0]],
      [[0, 1]],
    ], dtype=tf.float32)
    N = 2
    patches = extractor._extractPatches(data[:N], pos).numpy()
    tf.assert_equal(patches.shape, [N, 3, 3, 1 + 3])

    # first patch is at (0, 0) of first image
    x = patches[0, 1:, 1:, 0]
    expected = data[0, :2, :2, 0].numpy()
    assert np.allclose(x, expected)

    # second patch is at (0, 1) of second image
    x = patches[1, :2, 1:, 0]
    expected = data[1, -2:, :2, 0].numpy()
    assert np.allclose(x, expected)
    return

  ##################################
  test_patchMesh()
  test_patchIndices()
  test_coordGrid()
  
  test_pixelInfo()
  test_pixelMetrics_center()
  test_pixelMetrics_zero()

  test_extractPatchesRaw()
  test_extractPatches()
  test_extractPatches_independent()
  pass