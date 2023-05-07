import numpy as np
import tensorflow as tf
from NN.encoders.CConvExtractor import CConvExtractor, CBasicConvMixer

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