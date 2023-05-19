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

# test conversion from continuous to discrete time
def test_continuous_to_discrete():
  schedule = CDPDiscrete( beta_schedule=get_beta_schedule('linear'), noise_steps=10 )
  T = schedule.to_discrete( tf.linspace(0.0, 1.0, schedule.noise_steps) )
  tf.assert_equal(T, tf.range(schedule.noise_steps))
  tf.assert_equal(T[0], 0, 'first time step should be 0')
  tf.assert_equal(T[-1], schedule.noise_steps - 1, 'last time step should be noise_steps - 1')
  return

# test conversion from discrete to continuous time
def test_discrete_to_continuous():
  schedule = CDPDiscrete( beta_schedule=get_beta_schedule('linear'), noise_steps=10 )
  t = schedule.to_continuous( tf.range(schedule.noise_steps) )
  tf.assert_equal(t, tf.linspace(0.0, 1.0, schedule.noise_steps))
  tf.assert_equal(t[0], 0.0, 'first time step should be 0.0')
  tf.assert_equal(t[-1], 1.0, 'last time step should be 1.0')
  return