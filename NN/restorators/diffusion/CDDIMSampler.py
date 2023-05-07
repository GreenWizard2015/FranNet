import tensorflow as tf
from .IDiffusionSampler import IDiffusionSampler
from .diffusion_schedulers import CDiffusionParameters

# useful links:
#   https://github.com/cloneofsimo/minDiffusion/blob/master/mindiffusion/ddim.py
#   https://github.com/filipbasara0/simple-diffusion/blob/main/scheduler/ddim.py
#   https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py
# 
# "direction pointing to x_t" cause a problem, so I introduce "directionCoef" parameter to control it (original value is 1.0)
class CDDIMSampler(IDiffusionSampler):
  def __init__(self, stochasticity, directionCoef, noise_provider, steps):
    assert (0.0 <= stochasticity <= 1.0), 'Stochasticity must be in [0, 1] range'
    self._eta = stochasticity
    self._stepsConfig = steps
    self._directionCoef = directionCoef
    self._noise_provider = noise_provider
    return

  def _stepsSequence(self, startStep, endStep, kwargs={}):
    config = kwargs.get('steps skip type', self._stepsConfig)
    name = config['name'].lower() if isinstance(config, dict) else config.lower()
    if 'uniform' == name:
      K = config['K']
      steps = tf.range(endStep + 1, startStep - 1, K, dtype=tf.int32)[::-1]
      steps = tf.concat([[startStep - 1], steps], axis=0)
      prevSteps = tf.concat([steps[1:], [endStep]], axis=0)
      return steps, prevSteps
    
    if 'quadratic' == name:
      N = tf.cast(tf.abs(startStep - endStep), tf.float32)
      logN = tf.math.ceil(tf.math.log(N) / tf.math.log(2.0))
      logN = tf.cast(logN, tf.int32)
      steps = tf.range(logN, dtype=tf.int32)
      steps = endStep + tf.pow(2, steps)

      steps = tf.cast(steps, tf.int32)[::-1]
      steps = tf.concat([[startStep - 1], steps], axis=0)
      steps = tf.unique(steps).y # i'm too lazy to write proper code for this :D
      prevSteps = tf.concat([steps[1:], [endStep]], axis=0)
      return steps, prevSteps
    
    raise NotImplementedError('Unknown steps sequence name: {}'.format(name))
  
  def _get_variance(self, alpha_prod_t, alpha_prod_t_prev):
    beta_prod_t = 1.0 - alpha_prod_t
    beta_prod_t_prev = 1.0 - alpha_prod_t_prev
    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
    return variance
  
  def _reverseStep(self, model, schedule, eta, directionCoef):
    def f(x, t, tPrev):
      predictedNoise = model(x, t)
      # based on https://github.com/filipbasara0/simple-diffusion/blob/main/scheduler/ddim.py
      # obtain parameters for the current step and previous step
      (alpha_prod_t, ) = schedule.parametersForT(t, [ CDiffusionParameters.PARAM_ALPHA_HAT ])
      (alpha_prod_t_prev, ) = schedule.parametersForT(tPrev, [ CDiffusionParameters.PARAM_ALPHA_HAT ])

      std_dev_t = eta * tf.sqrt( self._get_variance(alpha_prod_t, alpha_prod_t_prev) )
      stochasticity_var = std_dev_t ** 2
      #######################################
      # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
      scaled_noise = tf.sqrt(1.0 - alpha_prod_t) * predictedNoise
      pred_original_sample = (x - scaled_noise) / tf.sqrt(alpha_prod_t)

      # compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
      coef2 = tf.sqrt(1.0 - alpha_prod_t_prev - stochasticity_var)
      coef2 = directionCoef * coef2 # <--- this is the only difference from the original formula
      
      # compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
      # NOTE: this is same as forward diffusion step (x0, x1, tPrev), but with "space" for noise (std_dev_t ** 2)
      coef1 = tf.sqrt(alpha_prod_t_prev)
      x_prev = (coef1 * pred_original_sample) + (coef2 * predictedNoise)
      return(x_prev, stochasticity_var)
    return f
  
  def sample(self, value, model, schedule, **kwargs):
    assert schedule.is_discrete, 'CDIMSampler supports only discrete schedules (for now)'
    maxSteps = schedule.noise_steps - 1
    startStep = kwargs.get('startStep', None) or maxSteps
    startStep = tf.clip_by_value(startStep, 0, maxSteps)

    endStep = kwargs.get('endStep', None) or 0
    endStep = tf.clip_by_value(endStep, 0, maxSteps)

    reverseStep = self._reverseStep(
      model,
      schedule=schedule,
      eta=kwargs.get('stochasticity', self._eta),
      directionCoef=kwargs.get('directionScale', self._directionCoef)
    ) # returns closure

    initShape = tf.shape(value)
    noise_provider = kwargs.get('noiseProvider', self._noise_provider)
    # TODO: fix _stepsSequence so we don't need to messing with +-1 here
    steps, prevSteps = self._stepsSequence(startStep + 1, endStep - 1, kwargs)
    for stepInd in tf.range(tf.size(steps)):
      step = steps[stepInd]
      prevStep = prevSteps[stepInd]
      value, variance = reverseStep(value, step, prevStep)
      value += noise_provider(initShape, variance)
      continue
    tf.assert_equal(tf.shape(value), initShape)
    return value
  pass
# End of CDDIMSampler