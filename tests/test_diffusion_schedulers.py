import tensorflow as tf
from NN.restorators.diffusion.diffusion_schedulers import CDPDiscrete, get_beta_schedule

def test_minus1_step():
  schedule = CDPDiscrete( beta_schedule=get_beta_schedule('linear'), noise_steps=10 )
  step = schedule.parametersForT([[-1]])
  tf.assert_equal(step['beta'], 0.0)
  tf.assert_equal(step['alpha'], 1.0)
  tf.assert_equal(step['alpha_hat'], 1.0)
  tf.assert_equal(step['posterior_variance'], 0.0)
  tf.assert_equal(step['SNR'], float('inf'))
  return

def test_monotonicity():
  schedule = CDPDiscrete( beta_schedule=get_beta_schedule('linear'), noise_steps=10 )
  steps = schedule.parametersForT(tf.range(-1, 10))
  # monotonic increase
  for name in ['beta', 'posterior_variance']:
    data = steps[name]
    tf.debugging.assert_greater_equal(data[1:], data[:-1], f'{name} is not monotonic')
    continue
  # monotonic decrease
  for name in ['alpha', 'alpha_hat', 'SNR']:
    data = steps[name]
    tf.debugging.assert_less_equal(data[1:], data[:-1], f'{name} is not monotonic')
    continue
  return