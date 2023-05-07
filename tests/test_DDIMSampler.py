import tensorflow as tf
from NN.restorators.diffusion.diffusion_samplers import sampler_from_config
import pytest

def make_sampler(config):
  return sampler_from_config({
    'name': 'ddim',
    'stochasticity': 0.0,
    'direction scale': 1.0,
    'noise stddev': 'zero',
    **config
  })

def _test_common(config):
  sampler = make_sampler(config)
  for i in range(3, 100):
    steps, prevSteps = sampler._stepsSequence(i, 0)
    tf.assert_equal(steps[0], [i - 1])
    tf.assert_equal(steps[-1], [1])
    tf.assert_equal(prevSteps[-1], [0])
    # steps should be strictly decreasing
    tf.assert_less(steps[1:], steps[:-1])
    tf.assert_less(prevSteps[1:], prevSteps[:-1])

    # steps and prevSteps should be equal except for the first element and the last element
    tf.assert_equal(steps[1:], prevSteps[:-1])

    # should be no duplicates
    tf.assert_equal(tf.size(steps), tf.size(tf.unique(steps).y))
    tf.assert_equal(tf.size(prevSteps), tf.size(tf.unique(prevSteps).y))
    continue
  return

@pytest.mark.parametrize('K', list(range(1, 10)))
def test_uniform_K_common(K):
  _test_common({ 'steps skip type': { 'name': 'uniform', 'K': K } })
  return

def test_quadratic_common():
  _test_common({ 'steps skip type': 'quadratic' })
  return

def test_uniform_steps():
  sampler = make_sampler({ 'steps skip type': { 'name': 'uniform', 'K': 3 } })
  steps, prevSteps = sampler._stepsSequence(10, 0)
  tf.assert_equal(steps, [9, 7, 4, 1])
  tf.assert_equal(prevSteps, [7, 4, 1, 0])
  return

def test_quadratic_steps_no_duplicate():
  sampler = make_sampler({ 'steps skip type': 'quadratic' })
  steps, prevSteps = sampler._stepsSequence(17, 0)
  tf.assert_equal(steps, [16, 8, 4, 2, 1])
  tf.assert_equal(prevSteps, [8, 4, 2, 1, 0])
  return

def test_quadratic_steps_case1():
  sampler = make_sampler({ 'steps skip type': 'quadratic' })
  steps, prevSteps = sampler._stepsSequence(21, 3)
  tf.assert_equal(steps, [20, 19, 11, 7, 5, 4])
  tf.assert_equal(prevSteps, [19, 11, 7, 5, 4, 3])
  return

def test_steps_K1():
  sampler = make_sampler({ 'steps skip type': { 'name': 'uniform', 'K': 1 } })
  steps, prevSteps = sampler._stepsSequence(10 + 1, 0 - 1)
  tf.assert_equal(steps, [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
  tf.assert_equal(prevSteps, [9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1])
  return
