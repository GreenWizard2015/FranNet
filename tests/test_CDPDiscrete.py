import tensorflow as tf
import numpy as np
import pytest
from NN.restorators.diffusion.diffusion_schedulers import get_beta_schedule, CDPDiscrete

# Test CDPDiscrete initialization
def test_CDPDiscrete_init():
  noise_steps = 10
  params = CDPDiscrete(beta_schedule=get_beta_schedule("linear"), noise_steps=noise_steps)
  assert params.noise_steps == noise_steps
  assert params.is_discrete == True
  return

# Test CDPDiscrete parametersForT method
def test_CDPDiscrete_parameters_for_t():
  params = CDPDiscrete(beta_schedule=get_beta_schedule("linear"), noise_steps=10)

  step = params.parametersForT([[5]])
  beta, alpha = step.beta, step.alpha
  tf.assert_equal(tf.shape(beta), tf.shape(alpha))
  tf.assert_equal(tf.shape(beta), (1, 1))

  step = params.parametersForT([[7]])
  posterior_variance, snr = step.posteriorVariance, step.SNR
  tf.assert_equal(tf.shape(posterior_variance), tf.shape(snr))
  tf.assert_equal(tf.shape(posterior_variance), (1, 1))
  return

# Test that posterior variance is same as calculated in DDIM sampler
@pytest.mark.parametrize("scheduler", ["linear", "cosine", "sigmoid", "quadratic"])
def test_CDPDiscrete_posterior_variance(scheduler):
  def _get_variance(alpha_prod_t, alpha_prod_t_prev):
    beta_prod_t = 1.0 - alpha_prod_t
    beta_prod_t_prev = 1.0 - alpha_prod_t_prev
    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
    return variance
  # end _get_variance
  params = CDPDiscrete(beta_schedule=get_beta_schedule(scheduler), noise_steps=100)

  T = tf.range(params.noise_steps)[..., None]
  step = params.parametersForT(T)
  posterior_variance, alpha_hat = step.posteriorVariance, step.alphaHat
  alpha_hat_prev = tf.concat([[[1.0]], alpha_hat[:-1]], axis=0)
  variance = _get_variance(alpha_hat, alpha_hat_prev).numpy()[..., 0]
  posterior_variance = posterior_variance.numpy()[..., 0]

  # compare variance and posterior_variance one by one
  for i, (v, pv) in enumerate(zip(variance, posterior_variance)):
    assert np.isclose(v, pv, atol=1e-6), f"i={i}, v={v}, pv={pv}"
  return