import tensorflow as tf
from NN.diffusion_schedulers import CDiffusionParameters

class IDiffusionSampler:
  def sample(self, *args, **kwargs):
    raise NotImplementedError()
  
class CDDPMSampler(IDiffusionSampler):
  def sample(self, value, model, schedule, startStep, endStep):
    initShape = tf.shape(value)

    def reverseStep(x, t):
      predictedNoise = model(x, t)
      # obtain parameters for the current step
      (variance, alpha, beta, alpha_hat) = schedule.parametersForT(t, [
        CDiffusionParameters.PARAM_POSTERIOR_VARIANCE,
        CDiffusionParameters.PARAM_ALPHA,
        CDiffusionParameters.PARAM_BETA,
        CDiffusionParameters.PARAM_ALPHA_HAT,
      ])
      # scale predicted noise
      # s = (1 - alpha) / sqrt(1 - alpha_hat)) = beta / sqrt(1 - alpha_hat)
      s = beta / tf.sqrt(1.0 - alpha_hat)
      x_prev = (x - s * predictedNoise) / tf.sqrt(alpha)
      tf.assert_equal(tf.shape(x_prev), initShape)
      # TODO: figure out why variance works better than sqrt(variance), as in the original paper
      return(x_prev, variance)
    ########################################
    if startStep is None:
      startStep = schedule.noise_steps - 1
    steps = tf.range(startStep, endStep, -1, dtype=tf.int32)
    for step in steps:
      value, variance = reverseStep(value, step)
      value += tf.random.normal(initShape) * variance
      continue
    tf.assert_equal(tf.shape(value), initShape)
    return value
  pass

# TODO: figure out why works worse than CDDPMSampler, even with K=1 and stochasticity=0.0 or 1.0
# when stochasticity=0.0 variance is always 0.0, so it is not the reason why it works worse
# usefull links:
#   https://github.com/cloneofsimo/minDiffusion/blob/master/mindiffusion/ddim.py
#   https://github.com/filipbasara0/simple-diffusion/blob/main/scheduler/ddim.py
class CDDIMSampler(IDiffusionSampler):
  def __init__(self, stochasticity, K):
    assert (0.0 <= stochasticity <= 1.0), 'Stochasticity must be in [0, 1] range'
    self._eta = stochasticity
    self._K = K
    return
  
  def sample(self, value, model, schedule, startStep, endStep):
    assert 0 == endStep, 'This DDIM sampler does not support endStep != 0'
    initShape = tf.shape(value)

    def get_variance(alpha_prod_t, alpha_prod_t_prev):
      beta_prod_t = 1.0 - alpha_prod_t
      beta_prod_t_prev = 1.0 - alpha_prod_t_prev
      variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
      return variance
    def reverseStep(x, t, tPrev):
      predictedNoise = model(x, t)
      # based on https://github.com/filipbasara0/simple-diffusion/blob/main/scheduler/ddim.py
      # obtain parameters for the current step and previous step
      (alpha_prod_t, ) = schedule.parametersForT(t, [ CDiffusionParameters.PARAM_ALPHA_HAT ])
      (alpha_prod_t_prev, ) = schedule.parametersForT(tPrev, [ CDiffusionParameters.PARAM_ALPHA_HAT ])

      variance = get_variance(alpha_prod_t, alpha_prod_t_prev)
      # compute stddev
      std_dev_t = self._stochasticity * tf.sqrt(variance)
      #######################################
      # compute predicted original sample from predicted noise also called
      # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
      scaled_noise = tf.sqrt(1.0 - alpha_prod_t) * predictedNoise
      pred_original_sample = (x - scaled_noise) / tf.sqrt(alpha_prod_t)

      # compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
      coef = tf.sqrt(1 - alpha_prod_t_prev - std_dev_t ** 2)
      pred_sample_direction = coef * predictedNoise
      
      # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
      x_prev = tf.sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction
      
      return(x_prev, std_dev_t)
    ########################################
    K = self._K
    if startStep is None:
      startStep = schedule.noise_steps

    # if startStep is 20, endStep is 0, and K is 5, then steps will be [16, 11, 6, 1]
    steps = tf.range(endStep + 1, startStep, K, dtype=tf.int32)[::-1]
    for step in steps:
      prevStep = tf.maximum(step - K, endStep)
      value, variance = reverseStep(value, step, prevStep)
      value += tf.random.normal(initShape) * variance
      continue
    tf.assert_equal(tf.shape(value), initShape)
    return value
  pass
  
def sampler_from_config(config):
  if isinstance(config, str) and ('ddpm' == config.lower()):
    return CDDPMSampler()
  
  if isinstance(config, dict) and ('ddim' == config['kind'].lower()):
    return CDDIMSampler(
      stochasticity=config['stochasticity'],
      K=config['K'],
    )
  
  raise ValueError('Unknown sampler: %s' % config)