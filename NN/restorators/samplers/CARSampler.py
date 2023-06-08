import tensorflow as tf
from Utils.utils import CFakeObject
from .CBasicInterpolantSampler import CBasicInterpolantSampler, ISamplingAlgorithm
from NN.restorators.samplers.steps_schedule import CProcessStepsDecayed

# TODO: adjust variance based on the "distance" between steps prevT and prevT - 1
class CARSamplingAlgorithm(ISamplingAlgorithm):
  def __init__(self, noiseProvider, steps, threshold):
    self._noiseProvider = noiseProvider
    self._steps = steps
    threshold = float(threshold)
    self._threshold = threshold if 0.0 < threshold else None
    return
  
  # crate a closure that will be used to update the state of the algorithm
  @staticmethod
  def _stateHandler(steps, threshold):
    if threshold is None:
      def stateF(current_step, **kwargs):
        active = current_step < steps.limit
        return dict(active=active)
      return stateF
    
    def stateF(current_step, mask, **kwargs):
      active = tf.logical_and(current_step < steps.limit, tf.reduce_any(mask))
      return dict(active=active, mask=mask)
    return stateF
  
  # crate a closure that will be used to update the value
  @staticmethod
  def _updateValueHandler(threshold):
    if threshold is None: # disable the thresholding
      return lambda x_prev, **kwargs: x_prev
    
    def updateValueF(x_prev, value, step, **kwargs):
      # restore the values that are already converged
      indices = tf.where(step.mask)
      return tf.tensor_scatter_nd_update(value, indices, x_prev)
    return updateValueF
  
  # crate a closure that will be used to get the current value
  @staticmethod
  def _currentValueHandler(threshold):
    if threshold is None: # disable the thresholding
      return lambda step: step.xt
    
    def currentValueF(step, **kwargs):
      return tf.boolean_mask(step.xt, step.mask, axis=0)
    return currentValueF
  
  def _makeStep(self, current_step, xt, params, **kwargs):
    params = CFakeObject(**params)
    stepV = params.steps.at(current_step, **kwargs)
    noise = params.noise_provider(shape=tf.shape(xt), sigma=tf.sqrt(stepV.variance))

    state = params.state(current_step=current_step, xt=xt, **kwargs)
    return CFakeObject(
      current_step=current_step,
      xt=xt + noise,
      **state
    )
  
  # remove 'iteration' from the kwargs to avoid conflicts with the 'iteration' argument in other methods
  def firstStep(self, value, iteration, **kwargs):
    B = tf.shape(value)[0]
    mask = tf.fill((B, ), True)
    # update kwargs, replace/store some data/callbacks
    noise_provider = kwargs.get('noiseProvider', self._noiseProvider)
    steps = kwargs.get('steps', self._steps)
    threshold = kwargs.get('threshold', self._threshold)

    kwargs = dict(
      **kwargs,
      params=dict(
        noise_provider=noise_provider,
        steps=steps,
        threshold=threshold,
        state=self._stateHandler(steps=steps, threshold=threshold),
        updateValue=self._updateValueHandler(threshold=threshold),
        currentValue=self._currentValueHandler(threshold=threshold),
      )
    )
    return self._makeStep(current_step=0, mask=mask, xt=value, **kwargs), kwargs
  
  def nextStep(self, step, value, solution, params, **kwargs):
    paramsRaw = params
    params = CFakeObject(**paramsRaw)
    # mask out the values that are already converged
    if not(params.threshold is None):
      diff = tf.abs(value - solution.value)
      kwargs = dict(
        **kwargs,
        mask=tf.reduce_any(params.threshold < diff, axis=-1)
      )

    return self._makeStep(
      current_step=step.current_step + 1,
      xt=solution.value,
      params=paramsRaw,
      **kwargs
    )

  def inference(self, model, step, interpolant, params, **kwargs):
    params = CFakeObject(**params)
    mask = None if params.threshold is None else step.mask
    stepV = params.steps.at(step.current_step, **kwargs)
    inference = interpolant.inference(xT=step.xt, T=stepV.T)
    return model(x=inference['xT'], t=inference['T'], mask=mask, **kwargs)
  
  def solve(self, x_hat, step, value, interpolant, params, **kwargs):
    params = CFakeObject(**params)
    xt = params.currentValue(step)
    tf.assert_equal(tf.shape(xt), tf.shape(x_hat))

    stepV = params.steps.at(step.current_step, **kwargs)
    solved = interpolant.solve(x_hat=x_hat, xt=xt, t=stepV.T)
    x_prev = interpolant.interpolate(x0=solved.x0, x1=solved.x1, t=stepV.prevT)
    tf.assert_equal(tf.shape(x_prev), tf.shape(x_hat))

    # return solution and additional information for debugging
    return CFakeObject(
      value=params.updateValue(x_prev=x_prev, value=value, step=step),
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