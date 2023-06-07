import tensorflow as tf
from Utils.utils import CFakeObject
from .CBasicInterpolantSampler import CBasicInterpolantSampler, ISamplingAlgorithm

# TODO: adjust variance based on the "distance" between steps prevT and prevT - 1
# TODO: find a way to jump to the T=0.0, if we converged or exceeded the steps limit
class CProcessStepsDecayed:
  def __init__(self, start, end, steps, decay):
    self._start = start
    self._end = end
    self._steps = steps
    self._decay = decay
    return
  
  def _at(self, step, **kwargs):
    step = tf.cast(step, tf.float32)
    current = tf.pow(self._decay, step) * self._start
    current = tf.clip_by_value(current, self._end, self._start)
    return current
  
  def at(self, step, **kwargs):
    current = self._at(step, **kwargs)
    prevT = self._at(step - 1, **kwargs)
    return CFakeObject(
      variance=prevT,
      T=current,
      prevT=prevT,
    )
  
  @property
  def limit(self): return self._steps
# End of CProcessSteps

class CARSamplingAlgorithm(ISamplingAlgorithm):
  def __init__(self, noiseProvider, steps, threshold):
    self._noiseProvider = noiseProvider
    self._steps = steps
    self._threshold = threshold
    return
  
  def _makeStep(self, current_step, xt, **kwargs):
    noise_provider = kwargs.get('noiseProvider', self._noiseProvider)
    stepV = self._steps.at(current_step, **kwargs)
    noise = noise_provider(shape=tf.shape(xt), sigma=tf.sqrt(stepV.variance))

    active = current_step < self._steps.limit
    args = {}
    threshold = kwargs.get('threshold', self._threshold)
    if not(threshold is None):
      mask = kwargs['mask']
      active = tf.logical_and(active, tf.reduce_any(mask))
      args = {**args, 'mask': mask}
      pass

    return CFakeObject(
      current_step=current_step,
      active=active,
      xt=xt + noise,
      **args
    )
  
  def firstStep(self, value, **kwargs):
    B = tf.shape(value)[0]
    mask = tf.fill((B, ), True)
    return self._makeStep(current_step=0, mask=mask, xt=value, **kwargs)
  
  def nextStep(self, step, value, solution, **kwargs):
    # mask out the values that are already converged
    threshold = kwargs.get('threshold', self._threshold)
    if not(threshold is None):
      mdiff = tf.reduce_max(tf.abs(value - solution.value), axis=-1)
      kwargs = {**kwargs, 'mask': threshold < mdiff}

    return self._makeStep(
      current_step=step.current_step + 1,
      xt=value,
      **kwargs
    )

  def inference(self, model, step, interpolant, **kwargs):
    threshold = kwargs.get('threshold', self._threshold)
    mask = None if threshold is None else step.mask
    stepV = self._steps.at(step.current_step, **kwargs)
    inference = interpolant.inference(xT=step.xt, T=stepV.T)
    return model(x=inference['xT'], t=inference['T'], mask=mask, **kwargs)
  
  def solve(self, x_hat, step, value, interpolant, **kwargs):
    stepV = self._steps.at(step.current_step, **kwargs)
    xt = step.xt
    threshold = kwargs.get('threshold', self._threshold)
    if not(threshold is None):
      xt = tf.boolean_mask(step.xt, step.mask, axis=0)
  
    tf.assert_equal(tf.shape(xt), tf.shape(x_hat))
    solved = interpolant.solve(x_hat=x_hat, xt=xt, t=stepV.T)
    x_prev = interpolant.interpolate(x0=solved.x0, x1=solved.x1, t=stepV.prevT)
    tf.assert_equal(tf.shape(x_prev), tf.shape(x_hat))

    if not(threshold is None):
      # restore the values that are already converged
      indices = tf.where(step.mask)
      x_prev = tf.tensor_scatter_nd_update(value, indices, x_prev)

    # return solution and additional information for debugging
    return CFakeObject(
      value=x_prev,
      x0=solved.x0,
      x1=solved.x1,
      current_step=step.current_step,
    )
# End of CARSamplingAlgorithm

class CARSampler(CBasicInterpolantSampler):
  def __init__(self, interpolant, noiseProvider, steps, threshold):
    super().__init__(
      interpolant=interpolant,
      algorithm=CARSamplingAlgorithm(
        noiseProvider=noiseProvider,
        steps=steps,
        threshold=threshold,
      )
    )
    return
  
  def train(self, x0, x1, T):
    return self._interpolant.train(x0=x0, x1=x1, T=T)
# End of CARSampler

def autoregressive_sampler_from_config(config):
  assert 'autoregressive' == config['name'].lower()

  from ..interpolants import interpolant_from_config
  from ..CNoiseProvider import noise_provider_from_config
  
  steps = config['steps']
  return CARSampler(
    interpolant=interpolant_from_config(config['interpolant']),
    noiseProvider=noise_provider_from_config(config['noise provider']),
    threshold=config['threshold'],
    steps=CProcessStepsDecayed(
      start=steps['start'],
      end=steps['end'],
      steps=steps['steps'],
      decay=steps['decay']
    )
  )