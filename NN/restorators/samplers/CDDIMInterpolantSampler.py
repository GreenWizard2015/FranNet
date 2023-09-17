import tensorflow as tf
from Utils.utils import CFakeObject
from NN.utils import normVec
from .CBasicInterpolantSampler import CBasicInterpolantSampler, ISamplingAlgorithm

class CDDIMSamplingAlgorithm(ISamplingAlgorithm):
  def __init__(self, stochasticity, noiseProvider, schedule, steps, clipping, projectNoise):
    self._stochasticity = stochasticity
    self._noiseProvider = noiseProvider
    self._schedule = schedule
    self._steps = steps
    self._clipping = clipping
    self._projectNoise = projectNoise
    return
  
  def _makeStep(self, current_step, steps, **kwargs):
    schedule = kwargs.get('schedule', self._schedule)
    eta = kwargs.get('eta', self._stochasticity)
    
    T = steps[0][current_step]
    alpha_hat_t = schedule.parametersForT(T).alphaHat
    prevStepInd = steps[1][current_step]
    alpha_hat_t_prev = schedule.parametersForT(prevStepInd).alphaHat

    stepVariance = schedule.varianceBetween(alpha_hat_t, alpha_hat_t_prev)
    sigma = tf.sqrt(stepVariance) * eta

    return CFakeObject(
      steps=steps,
      current_step=current_step,
      active=(0 <= current_step),
      sigma=sigma,
      #
      T=T,
      t=alpha_hat_t,
      t_prev=alpha_hat_t_prev,
      t_prev_2=1.0 - alpha_hat_t_prev - tf.square(sigma),
    )
  
  def firstStep(self, **kwargs):
    schedule = kwargs.get('schedule', self._schedule)
    assert schedule is not None, 'schedule is None'
    assert schedule.is_discrete, 'schedule is not discrete'
    steps = schedule.steps_sequence(
      startStep=kwargs.get('startStep', None),
      endStep=kwargs.get('endStep', None),
      config=kwargs.get('stepsConfig', self._steps),
      reverse=True, # reverse steps order to make it easier to iterate over them
    )

    return self._makeStep(
      current_step=tf.size(steps[0]) - 1,
      steps=steps,
      **kwargs
    )
  
  def nextStep(self, step, **kwargs):
    return self._makeStep(
      current_step=step.current_step - 1,
      steps=step.steps,
      **kwargs
    )
  
  def inference(self, model, step, value, **kwargs):
    schedule = kwargs.get('schedule', self._schedule)
    return model(
      x=value,
      T=step.T,
      t=schedule.to_continuous(step.T),
    )
  
  def _withNoise(self, value, sigma, x0, kwargs):
    noise_provider = kwargs.get('noiseProvider', self._noiseProvider)
    noise = noise_provider(shape=tf.shape(value), sigma=sigma)
    if not kwargs.get('projectNoise', self._projectNoise): return value + noise
    
    _, L = normVec(value - x0)
    vec, _ = normVec(value + noise - x0)
    return x0 + L * vec # project noise back to the spherical manifold
  
  def _withClipping(self, value, kwargs):
    clipping = kwargs.get('clipping', self._clipping)
    if clipping is None: return value
    return tf.clip_by_value(value, clip_value_min=clipping['min'], clip_value_max=clipping['max'])

  def solve(self, x_hat, step, value, interpolant, **kwargs):
    solved = interpolant.solve(x_hat=x_hat, xt=value, t=step.t)
    x_prev = interpolant.interpolate(
      x0=solved.x0, x1=solved.x1,
      t=step.t_prev, t2=step.t_prev_2
    )
    x_prev = self._withNoise(x_prev, sigma=step.sigma, x0=solved.x0, kwargs=kwargs)
    x_prev = self._withClipping(x_prev, kwargs=kwargs)
    # return solution and additional information for debugging
    return CFakeObject(
      value=x_prev,
      x0=solved.x0,
      x1=solved.x1,
      T=step.T,
      current_step=step.current_step,
      sigma=step.sigma,
    )
# End of CDDIMSamplingAlgorithm

class CDDIMInterpolantSampler(CBasicInterpolantSampler):
  def __init__(
    self, interpolant,
    stochasticity, noiseProvider, schedule, steps, clipping, projectNoise
  ):
    super().__init__(
      interpolant=interpolant,
      algorithm=CDDIMSamplingAlgorithm(
        stochasticity=stochasticity,
        noiseProvider=noiseProvider,
        schedule=schedule,
        steps=steps,
        clipping=clipping,
        projectNoise=projectNoise
      )
    )
    self._schedule = schedule
    return
  
  def train(self, x0, x1, T):
    T = self._schedule.to_discrete(T)
    # apply training procedure from interpolant
    alpha_hat_t = self._schedule.parametersForT(T[:, 0]).alphaHat
    trainData = self._interpolant.train(x0=x0, x1=x1, T=alpha_hat_t)
    return {
      **trainData,
      'T': self._schedule.to_continuous(T)
    }
# End of CDDIMInterpolantSampler