import tensorflow as tf
from NN.utils import sample_halton_sequence

def _structuredSample(B, N, sharedShifts):
  # make a uniform grid of points (total N points)
  n = tf.cast(tf.math.ceil(tf.math.sqrt(tf.cast(N, tf.float32))), tf.int32)
  rng = tf.linspace(0.0, 1.0, n + 1)[:-1] # [0, 1) range
  delta = tf.abs(rng[1] - rng[0])
  x, y = tf.meshgrid(rng, rng)
  grid = tf.stack([tf.reshape(x, (-1, )), tf.reshape(y, (-1, ))], axis=0)
  grid = tf.transpose(grid)[None]
  tf.assert_equal(tf.shape(grid), (1, N, 2))
  # generate samples
  noiseShape = (B, 1, 2) if sharedShifts else (B, N, 2)
  shifts = tf.random.uniform(noiseShape) * delta
  res = shifts + tf.tile(grid, [B, 1, 1])
  # some verifications
  tf.assert_equal(tf.shape(res), (B, N, 2))
  tf.debugging.assert_greater_equal(res, 0.0)
  tf.debugging.assert_less_equal(res, 1.0)
  return res

def PositionsSampler(sampler):
  if callable(sampler): return sampler
  
  samplers = {
    'uniform': tf.random.uniform,
    'halton': lambda shape: sample_halton_sequence(shape[:-1], shape[-1]),
    'structured': lambda shape: _structuredSample(B=shape[0], N=shape[1], sharedShifts=True),
    'structured noisy': lambda shape: _structuredSample(B=shape[0], N=shape[1], sharedShifts=False)
  }
  assert sampler in samplers, f'Unknown training sampler ({sampler})'
  return samplers[sampler]