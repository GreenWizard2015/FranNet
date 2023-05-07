import pytest
import numpy as np
import tensorflow as tf
from NN.utils import extractInterpolated, flatCoordsGridTF

@pytest.mark.parametrize('hw', list(range(3, 16)))
def test_extractInterpolated(hw):
  x = np.random.uniform(size=(1, hw, hw, 1)).astype(np.float32)
  d = (1.0 / hw) / 2.0
  # check that boundaries are not at d
  a = extractInterpolated(x, np.array([[[0.0, 0.0]]]).astype(np.float32)).numpy()
  b = extractInterpolated(x, np.array([[[d, d]]]).astype(np.float32)).numpy()
  assert not np.allclose(a, b)
  # 
  for i in range(hw):
    for j in range(hw):
      pos = np.array([[[1 + i * 2, 1 + j * 2]]]) * d
      y = extractInterpolated(x, pos).numpy()
      target = x[0, j, i, 0]
      assert np.allclose(y, target, atol=1e-4), '%s != %s' % (y, target)
      continue
    continue
  
  pos = np.array([[[2 * d, 2 * d]]])
  y = extractInterpolated(x, pos).numpy()
  assert np.allclose(y, x[0, :2, :2, 0].mean())
  return

def test_flatCoordsGridTF():
  grid = flatCoordsGridTF(3).numpy()
  d = (1.0 / 3) / 2.0
  target = np.array([
    [d, d], [(1 + 2) * d, d], [(1 + 4) * d, d],
    [d, (1 + 2) * d], [(1 + 2) * d, (1 + 2) * d], [(1 + 4) * d, (1 + 2) * d],
    [d, (1 + 4) * d], [(1 + 2) * d, (1 + 4) * d], [(1 + 4) * d, (1 + 4) * d],
  ])
  assert np.allclose(grid, target)
  return
