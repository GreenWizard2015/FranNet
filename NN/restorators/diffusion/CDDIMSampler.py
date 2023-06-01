import tensorflow as tf
from .IDiffusionSampler import IDiffusionSampler
from ..common import clipping_from_config
from NN.utils import normVec
from Utils.utils import CFakeObject

# useful links:
#   https://github.com/cloneofsimo/minDiffusion/blob/master/mindiffusion/ddim.py
#   https://github.com/filipbasara0/simple-diffusion/blob/main/scheduler/ddim.py
#   https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py
class CDDIMSampler(IDiffusionSampler):
  def __init__(self, stochasticity, noise_provider, steps, clipping, projectNoise, useFloat64=False):
    assert (0.0 <= stochasticity <= 1.0), 'Stochasticity must be in [0, 1] range'
    self._eta = stochasticity
    self._stepsConfig = steps
    self._noise_provider = noise_provider
    self._clipping = clipping
    self._projectNoise = projectNoise
    self._useFloat64 = useFloat64
    return

  def _reverseStep_float64(self, model, schedule, eta):
    # use float64 and some tricks to improve numerical stability
    def f(x, t, tPrev):
      predictedNoise = model(x, t)
      # based on https://github.com/filipbasara0/simple-diffusion/blob/main/scheduler/ddim.py
      # obtain parameters for the current step and previous step
      t = schedule.parametersForT(t, dtype=tf.float64)
      tPrev = schedule.parametersForT(tPrev, dtype=tf.float64)

      stepVariance = schedule.varianceBetween(t.alphaHat, tPrev.alphaHat)
      sigma = tf.sqrt(stepVariance) * tf.cast(eta, dtype=stepVariance.dtype)
      #######################################
      noise_scale = tf.sqrt(1.0 - t.alphaHat)
      coef2 = tf.sqrt(1.0 - tPrev.alphaHat - tf.square(sigma))
      coef1 = tf.sqrt(tPrev.alphaHat / t.alphaHat)
      # convert all tensors to x.dtype
      coef1 = tf.cast(coef1, dtype=x.dtype)
      coef2 = tf.cast(coef2, dtype=x.dtype)
      noise_scale = tf.cast(noise_scale, dtype=x.dtype)
      sigma = tf.cast(sigma, dtype=x.dtype)
      
      x_minus_noise = x - (noise_scale * predictedNoise)
      x_prev = ( (coef1 * x_minus_noise) + (coef2 * predictedNoise) )
      tf.assert_equal(x.dtype, x_prev.dtype)
      x_prev = tf.ensure_shape(x_prev, x.shape)
      x0 = x_minus_noise / tf.cast(tf.sqrt(t.alphaHat), dtype=x.dtype)
      return CFakeObject(x_prev=x_prev, sigma=sigma, x0=x0, x1=predictedNoise)
    return f
  
  def _reverseStep_float32(self, model, schedule, eta):
    def f(x, t, tPrev):
      predictedNoise = model(x, t)
      isEndOfDiffusion = tf.equal(t, -1)
      # based on https://github.com/filipbasara0/simple-diffusion/blob/main/scheduler/ddim.py
      # obtain parameters for the current step and previous step
      t = schedule.parametersForT(t)
      tPrev = schedule.parametersForT(tPrev)

      stepVariance = schedule.varianceBetween(t.alphaHat, tPrev.alphaHat)
      sigma = tf.sqrt(stepVariance) * eta
      #######################################
      # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
      scaled_noise = t.sqrt_one_minus_alpha_hat * predictedNoise
      pred_original_sample = (x - scaled_noise) / t.sqrt_alpha_hat

      # compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
      coef2 = tf.sqrt(1.0 - tPrev.alphaHat - tf.square(sigma))
      
      # compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
      # NOTE: this is same as forward diffusion step (x0, x1, tPrev), but with "space" for noise (std_dev_t ** 2)
      coef1 = tPrev.sqrt_alpha_hat
      x_prev = (coef1 * pred_original_sample) + (coef2 * predictedNoise)
      return CFakeObject(x_prev=x_prev, sigma=sigma, x0=pred_original_sample, x1=predictedNoise)
    return f
  
  def _reverseStep(self, model, schedule, eta):
    if self._useFloat64:
      return self._reverseStep_float64(model, schedule, eta)
    
    return self._reverseStep_float32(model, schedule, eta)
  
  def _valueUpdater(self, noise_provider, projectNoise):
    if not projectNoise: # just add noise to the new value
      return lambda step: step.x_prev + noise_provider(shape=tf.shape(step.x_prev), sigma=step.sigma)
    # with noise projection
    def f(step):
      _, L = normVec(step.x_prev - step.x0)
      value = step.x_prev + noise_provider(shape=tf.shape(step.x_prev), sigma=step.sigma)
      vec, _ = normVec(value - step.x0)
      return step.x0 + L * vec # project noise back to the spherical manifold
    return f

  def sample(self, value, model, schedule, **kwargs):
    assert schedule.is_discrete, 'CDDIMSampler supports only discrete schedules (for now)'
    steps, prevSteps = schedule.steps_sequence(
      startStep=kwargs.get('startStep', None),
      endStep=kwargs.get('endStep', None),
      config=kwargs.get('stepsConfig', self._stepsConfig),
    )
    
    reverseStep = self._reverseStep(
      model,
      schedule=schedule,
      eta=kwargs.get('stochasticity', self._eta),
    ) # returns closure

    initShape = tf.shape(value)
    updateValue = self._valueUpdater(
      noise_provider=kwargs.get('noiseProvider', self._noise_provider),
      projectNoise=kwargs.get('projectNoise', self._projectNoise),
    )
    clipping = clipping_from_config(kwargs.get('clipping', self._clipping))
    value = clipping(value)
    for stepInd in tf.range(tf.size(steps)):
      rev = reverseStep(value, steps[stepInd], prevSteps[stepInd]) # perform reverse step
      value = updateValue(rev) # update value
      # clip value after updating it
      value = clipping(value)
      continue
    tf.assert_equal(tf.shape(value), initShape)
    return value
  pass
# End of CDDIMSampler