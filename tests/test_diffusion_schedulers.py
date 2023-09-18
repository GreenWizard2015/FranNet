import tensorflow as tf
import pytest
import numpy as np
from NN.restorators.diffusion.diffusion_schedulers import CDPDiscrete, get_beta_schedule

def test_minus1_step():
  schedule = CDPDiscrete( beta_schedule=get_beta_schedule('linear'), noise_steps=10 )
  step = schedule.parametersForT([[-1]])
  tf.assert_equal(step.beta, 0.0)
  tf.assert_equal(step.alpha, 1.0)
  tf.assert_equal(step.alphaHat, 1.0)
  tf.assert_equal(step.posteriorVariance, 0.0)
  tf.assert_equal(step.SNR, float('inf'))
  return

def test_monotonicity():
  schedule = CDPDiscrete( beta_schedule=get_beta_schedule('linear'), noise_steps=10 )
  steps = schedule.parametersForT(tf.range(-1, 10))
  # monotonic increase
  tf.debugging.assert_greater_equal(steps.beta[1:], steps.beta[:-1], 'beta is not monotonic')
  tf.debugging.assert_greater_equal(steps.posteriorVariance[1:], steps.posteriorVariance[:-1], 'posteriorVariance is not monotonic')
  
  # monotonic decrease
  tf.debugging.assert_less_equal(steps.alpha[1:], steps.alpha[:-1], 'alpha is not monotonic')
  tf.debugging.assert_less_equal(steps.alphaHat[1:], steps.alphaHat[:-1], 'alphaHat is not monotonic')
  tf.debugging.assert_less_equal(steps.SNR[1:], steps.SNR[:-1], 'SNR is not monotonic')
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

# posterior variance should be equal to varianceBetween
@pytest.mark.parametrize('schedule', ['linear', 'quadratic', 'sigmoid', 'cosine'])
def test_posterior_variance(schedule):
  schedule = CDPDiscrete( beta_schedule=get_beta_schedule(schedule), noise_steps=10 )
  steps = schedule.parametersForT(tf.range(-1, 10))
  t = steps.alphaHat
  var = steps.posteriorVariance[1:]
  varianceBetween = schedule.varianceBetween(t[1:], t[:-1])
  tf.debugging.assert_near(var, varianceBetween, atol=1e-6)
  return

# verify that posterior variance is same as in https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/annotated_diffusion.ipynb#scrollTo=f7628fb3
def test_posterior_variance_HF():
  extractedPosteriorVariance = [
    0.000000000000, 0.000403486629, 0.000836733670, 0.001288122730, 0.001750515075, 0.002220379887, 
    0.002695692703, 0.003175399266, 0.003659132170, 0.004146338440, 0.004637072328, 0.005131179933, 
    0.005629014224, 0.006130410824, 0.006635569502, 0.007144805044, 0.007658232935, 0.008176232688, 
    0.008699111640, 0.009226742201, 0.009759841487, 0.010298749432, 0.010843393393, 0.011394466273, 
    0.011952218600, 0.012517100200, 0.013089468703, 0.013669779524, 0.014258250594, 0.014855704270, 
    0.015462197363, 0.016078939661, 0.016705634072, 0.017343325540, 0.017992677167, 0.018653955311, 
    0.019328247756, 0.020015781745, 0.020717978477, 0.021434905007, 0.022167997435, 0.022917652503, 
    0.023685265332, 0.024471685290, 0.025277702138, 0.026105124503, 0.026954751462, 0.027827847749, 
    0.028726227582, 0.029651055112, 0.030604440719, 0.031587723643, 0.032603222877, 0.033652618527, 
    0.034738678485, 0.035863354802, 0.037030287087, 0.038241039962, 0.039500311017, 0.040810417384, 
    0.042176153511, 0.043600667268, 0.045090332627, 0.046648778021, 0.048282265663, 0.049998655915, 
    0.051801901311, 0.053703710437, 0.055709652603, 0.057833313942, 0.060083035380, 0.062474120408, 
    0.065019220114, 0.067738182843, 0.070647582412, 0.073771975935, 0.077136911452, 0.080773383379, 
    0.084718026221, 0.089011870325, 0.093707941473, 0.098867602646, 0.104565657675, 0.110894180834, 
    0.117968119681, 0.125927925110, 0.134962469339, 0.145302027464, 0.157262265682, 0.171261221170, 
    0.187877282500, 0.207921773195, 0.232590973377, 0.263695180416, 0.304120719433, 0.358746767044, 
    0.436440110207, 0.554701685905, 0.749393582344, 0.999657154083
  ]
  noise_steps = len(extractedPosteriorVariance)
  extractedPosteriorVariance = tf.convert_to_tensor(extractedPosteriorVariance, dtype=tf.float32)
  schedule = CDPDiscrete( beta_schedule=get_beta_schedule('cosine'), noise_steps=noise_steps )
  var = schedule.parametersForT(tf.range(noise_steps)).posteriorVariance
  tf.debugging.assert_near(var, extractedPosteriorVariance, atol=2e-6)
  return

# test conversion with lastStep=True
def test_last_step():
  schedule = CDPDiscrete( beta_schedule=get_beta_schedule('linear'), noise_steps=10 )
  N = 3
  t = tf.linspace(0.0, 1.0, schedule.noise_steps * N)
  tf.assert_equal(t[0], 0.0)
  tf.assert_equal(t[-1], 1.0)
  T = schedule.to_discrete(t, lastStep=True )
  expected = sum([ [i] * N for i in range(schedule.noise_steps) ], [])
  tf.assert_equal(T, expected)
  return