import tensorflow as tf
from .IDiffusionSampler import IDiffusionSampler
from ..common import clipping_from_config
from Utils.utils import CFakeObject

class CDDPMSampler(IDiffusionSampler):
  def __init__(self, noise_provider, clipping):
    self._noise_provider = noise_provider
    self._clipping = clipping
    return

  def _reverseStep(self, model, schedule):
    def reverseStep(x, t):
      predictedNoise = model(x, t)
      # obtain parameters for the current step
      currentStep = schedule.parametersForT(t)
      # scale predicted noise
      # NOTE: rollbacks only a single step of the diffusion process
      s = (1 - currentStep.alpha) / tf.sqrt(1.0 - currentStep.alphaHat)
      d = tf.sqrt(currentStep.alpha)
      # prevent NaNs/Infs
      s = tf.where(tf.math.is_finite(s), s, 0.0)
      d = tf.where(tf.math.is_finite(d), d, 1.0)
      d = tf.where(0.0 == d, 1.0, d)

      x_prev = (x - s * predictedNoise) / d
      return CFakeObject(x_prev=x_prev, sigma=currentStep.sigma)
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
    clipping = clipping_from_config(kwargs.get('clipping', self._clipping))
    value = clipping(value)
    for step in steps:
      rev = reverseStep(value, step)
      value = rev.x_prev + noise_provider(shape=initShape, sigma=rev.sigma)
      value = clipping(value)
      continue
    tf.assert_equal(tf.shape(value), initShape)
    return value
  pass
# end of class CDDPMSampler
