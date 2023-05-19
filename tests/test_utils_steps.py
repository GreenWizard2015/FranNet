import tensorflow as tf
from NN.utils import make_steps_sequence
import pytest

def _test_common(config):
  for i in range(3, 100):
    steps, prevSteps = make_steps_sequence(startStep=i, endStep=0, config=config)
    tf.assert_equal(tf.shape(steps), tf.shape(prevSteps))
    # steps should be strictly decreasing
    tf.assert_less(steps[1:], steps[:-1], message='Steps are not strictly decreasing')
    tf.assert_less(prevSteps[1:], prevSteps[:-1], message='PrevSteps are not strictly decreasing')

    tf.assert_equal(steps[0], [i - 1], message='First step is not i - 1')
    
    tf.assert_equal(steps[-1], [1], message='Last step is not 1')
    tf.assert_equal(prevSteps[-1], [0], message='Last prevStep is not 0')
    
    tf.assert_equal(steps[-2], [2], message='Second to last step is not 2')
    tf.assert_equal(prevSteps[-2], [1], message='Second to last prevStep is not 1')

    tf.assert_equal(steps[1:], prevSteps[:-1], message='Steps and prevSteps are not equal except for the first element and the last element')
    tf.assert_equal(tf.size(steps), tf.size(tf.unique(steps).y), message='Steps has duplicates')
    tf.assert_equal(tf.size(prevSteps), tf.size(tf.unique(prevSteps).y), message='PrevSteps has duplicates')
    continue
  return

@pytest.mark.parametrize('K', list(range(1, 10)))
def test_uniform_K_common(K):
  _test_common({ 'name': 'uniform', 'K': K })
  return

def test_quadratic_common():
  _test_common( 'quadratic' )
  return

def test_uniform_steps():
  config = { 'name': 'uniform', 'K': 3 }
  steps, prevSteps = make_steps_sequence(10, 0, config=config)
  tf.assert_equal(steps, [9, 2 + 3 + 3, 2 + 3, 2, 1])
  tf.assert_equal(prevSteps, [2 + 3 + 3, 2 + 3, 2, 1, 0])
  return

def test_quadratic_steps_no_duplicate():
  steps, prevSteps = make_steps_sequence(17, 0, config='quadratic')
  tf.assert_equal(steps, [16, 8, 4, 2, 1])
  tf.assert_equal(prevSteps, [8, 4, 2, 1, 0])
  return

def test_quadratic_steps_case1():
  steps, prevSteps = make_steps_sequence(21, 3, config='quadratic')
  tf.assert_equal(steps, [20, 19, 11, 7, 5, 4])
  tf.assert_equal(prevSteps, [19, 11, 7, 5, 4, 3])
  return

def test_steps_K1():
  config = { 'name': 'uniform', 'K': 1 }
  steps, prevSteps = make_steps_sequence(10 + 1, 0 - 1, config=config)
  tf.assert_equal(steps, [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
  tf.assert_equal(prevSteps, [9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1])
  return
