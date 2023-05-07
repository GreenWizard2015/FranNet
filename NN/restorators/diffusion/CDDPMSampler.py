import tensorflow as tf
from .IDiffusionSampler import IDiffusionSampler
from .diffusion_schedulers import CDiffusionParameters

class CDDPMSampler(IDiffusionSampler):
  def __init__(self, noise_provider):
    self._noise_provider = noise_provider
    return

  def _reverseStep(self, model, schedule):
    def reverseStep(x, t):
      predictedNoise = model(x, t)
      # obtain parameters for the current step
      (variance, alpha, alpha_hat) = schedule.parametersForT(t, [
        CDiffusionParameters.PARAM_POSTERIOR_VARIANCE,
        CDiffusionParameters.PARAM_ALPHA,
        CDiffusionParameters.PARAM_ALPHA_HAT,
      ])
      # scale predicted noise
      # NOTE: rollbacks only a single step of the diffusion process
      s = (1 - alpha) / tf.sqrt(1.0 - alpha_hat)
      d = tf.sqrt(alpha)
      # prevent NaNs/Infs
      s = tf.where(tf.math.is_finite(s), s, 0.0)
      d = tf.where(tf.math.is_finite(d), d, 1.0)
      d = tf.where(0.0 == d, 1.0, d)

      x_prev = (x - s * predictedNoise) / d
      return(x_prev, variance)
    return reverseStep
  
  def _stepsSequence(self, startStep, endStep, totalSteps):
    startStep = totalSteps if startStep is None else startStep
    steps = tf.range(startStep, endStep, -1, dtype=tf.int32)
    return steps - 1
  
  def sample(self, value, model, schedule, startStep, endStep, **kwargs):
    assert schedule.is_discrete, 'CDDPMSampler supports only discrete schedules (for now)'
    initShape = tf.shape(value)
    steps = self._stepsSequence(startStep, endStep, schedule.noise_steps)
    reverseStep = self._reverseStep(model, schedule) # returns closure
    noise_provider = kwargs.get('noise provider', self._noise_provider)
    for step in steps:
      value, variance = reverseStep(value, step)
      value += noise_provider(initShape, variance)
      continue
    tf.assert_equal(tf.shape(value), initShape)
    return value
  pass
# end of class CDDPMSampler
