import tensorflow as tf
from .IDiffusionSampler import IDiffusionSampler
from .diffusion_schedulers import CDiffusionParameters

class CDDPMSampler(IDiffusionSampler):
  def __init__(self, noise_provider, clipping):
    self._noise_provider = noise_provider
    self._clipping = clipping
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
  
  def sample(self, value, model, schedule, **kwargs):
    assert schedule.is_discrete, 'CDDPMSampler supports only discrete schedules (for now)'
    steps, _ = schedule.steps_sequence(
      startStep=kwargs.get('startStep', None),
      endStep=kwargs.get('endStep', None),
      config={ 'name': 'uniform', 'K': 1} # no skipping steps
    )
    
    initShape = tf.shape(value)
    reverseStep = self._reverseStep(model, schedule) # returns closure
    noise_provider = kwargs.get('noiseProvider', self._noise_provider)
    clippingArgs = kwargs.get('clipping', self._clipping)
    if clippingArgs is None:
      clipping = lambda x: x
    else:
      clipping = lambda x: tf.clip_by_value(x, clip_value_min=clippingArgs['min'], clip_value_max=clippingArgs['max'])

    value = clipping(value)
    for step in steps:
      value, variance = reverseStep(value, step)
      value += noise_provider(initShape, variance)
      value = clipping(value)
      continue
    tf.assert_equal(tf.shape(value), initShape)
    return value
  pass
# end of class CDDPMSampler
