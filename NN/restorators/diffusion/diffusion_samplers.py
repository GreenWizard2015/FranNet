import tensorflow as tf
from .diffusion_schedulers import CDiffusionParameters

# diffusion works better with squared/zero variance, but it is incorrect implementation, so I added this configurable option
class CNoiseProvider:
  def __init__(self, stddev_type):
    assert (stddev_type in ['correct', 'squared', 'zero']), 'Unknown variance type: {}'.format(stddev_type)

    self._generate_noise = lambda shape, variance: tf.random.normal(shape, stddev=tf.sqrt(variance))

    if 'squared' == stddev_type:
      self._generate_noise = lambda shape, variance: tf.random.normal(shape, stddev=variance)

    if 'zero' == stddev_type:
      self._generate_noise = lambda shape, variance: tf.zeros(shape, dtype=tf.float32)
    return
  
  def __call__(self, shape, variance):
    return self._generate_noise(shape, variance)
# end of class CNoiseProvider

class IDiffusionSampler:
  def sample(self, *args, **kwargs):
    raise NotImplementedError()
# end of class IDiffusionSampler

class CDDPMSampler(IDiffusionSampler):
  def __init__(self, noise_provider):
    self._noise_provider = noise_provider
    return

  def _reverseStep(self, model, schedule):
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

  def _stepsSequence(self, startStep, endStep, totalSteps, **kwargs):
    startStep = totalSteps - 1 if startStep is None else startStep

    config = kwargs.get('steps skip type', self._stepsConfig)
    kind = config['kind'].lower() if isinstance(config, dict) else config.lower()
    if 'uniform' == kind:
      K = config['K']
      steps = tf.range(endStep + 1, startStep - 1, K, dtype=tf.int32)[::-1]
      steps = tf.concat([[startStep - 1], steps], axis=0)
      prevSteps = tf.concat([steps[1:], [endStep]], axis=0)
      return steps, prevSteps
    
    if 'quadratic' == kind:
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
    
    raise NotImplementedError('Unknown steps sequence kind: {}'.format(kind))
  
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

      variance = self._get_variance(alpha_prod_t, alpha_prod_t_prev)
      std_dev_t = eta * tf.sqrt(variance)
      #######################################
      # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
      scaled_noise = tf.sqrt(1.0 - alpha_prod_t) * predictedNoise
      pred_original_sample = (x - scaled_noise) / tf.sqrt(alpha_prod_t)

      # compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
      coef = tf.sqrt(1.0 - alpha_prod_t_prev - std_dev_t ** 2)
      coef = directionCoef * coef # <--- this is the only difference from the original formula
      pred_sample_direction = coef * predictedNoise
      
      # compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
      x_prev = tf.sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction
      return(x_prev, std_dev_t ** 2)
    return f
  
  def sample(self, value, model, schedule, startStep, endStep, **kwargs):
    reverseStep = self._reverseStep(
      model,
      schedule=kwargs.get('schedule', schedule),
      eta=kwargs.get('stochasticity', self._eta),
      directionCoef=kwargs.get('direction scale', self._directionCoef)
    ) # returns closure

    initShape = tf.shape(value)
    noise_provider = kwargs.get('noise provider', self._noise_provider)
    steps, prevSteps = self._stepsSequence(startStep, endStep, schedule.noise_steps, kwargs)
    for stepInd in tf.range(tf.size(steps)):
      step = steps[stepInd]
      prevStep = prevSteps[stepInd]
      value, variance = reverseStep(value, step, prevStep)
      value += noise_provider(initShape, variance)
      continue
    tf.assert_equal(tf.shape(value), initShape)
    return value
  pass
  
def noise_provider_from_config(config):
  if isinstance(config, str):
    return CNoiseProvider(stddev_type=config)
  
  raise ValueError('Unknown noise provider: %s' % config)

def sampler_from_config(config):
  if isinstance(config, dict) and ('ddpm' == config['kind'].lower()):
    return CDDPMSampler(
      noise_provider=noise_provider_from_config(config['noise stddev']),
    )
  
  if isinstance(config, dict) and ('ddim' == config['kind'].lower()):
    return CDDIMSampler(
      stochasticity=config['stochasticity'],
      directionCoef=config['direction scale'],
      noise_provider=noise_provider_from_config(config['noise stddev']),
      steps=config['steps skip type'],
    )
  
  raise ValueError('Unknown sampler: %s' % config)

if __name__ == '__main__': # very dumb tests
  def DDIMTests():
    DDIM_CONFIG = {
      'kind': 'ddim',
      'stochasticity': 0.1,
      'direction scale': 1.0,
      'noise stddev': 'zero',
    }
    def make_sampler(config):
      return sampler_from_config(dict(DDIM_CONFIG, **config))
    
    def test_common(config):
      sampler = make_sampler(config)
      for i in range(3, 100):
        steps, prevSteps = sampler._stepsSequence(i, 0, None)
        tf.assert_equal(steps[0], [i - 1])
        tf.assert_equal(steps[-1], [1])
        tf.assert_equal(prevSteps[-1], [0])
        # steps should be strictly decreasing
        tf.assert_less(steps[1:], steps[:-1])
        tf.assert_less(prevSteps[1:], prevSteps[:-1])

        # steps and prevSteps should be equal except for the first element and the last element
        tf.assert_equal(steps[1:], prevSteps[:-1])

        # should be no duplicates
        tf.assert_equal(tf.size(steps), tf.size(tf.unique(steps).y))
        tf.assert_equal(tf.size(prevSteps), tf.size(tf.unique(prevSteps).y))
        continue
      return
    
    def test_uniform_steps():
      sampler = make_sampler({ 'steps skip type': { 'kind': 'uniform', 'K': 3 } })
      steps, prevSteps = sampler._stepsSequence(10, 0, 10)
      tf.assert_equal(steps, [9, 7, 4, 1])
      tf.assert_equal(prevSteps, [7, 4, 1, 0])
      return
    
    def test_quadratic_steps_no_duplicate():
      sampler = make_sampler({ 'steps skip type': 'quadratic' })
      steps, prevSteps = sampler._stepsSequence(17, 0, None)
      tf.assert_equal(steps, [16, 8, 4, 2, 1])
      tf.assert_equal(prevSteps, [8, 4, 2, 1, 0])
      return
    
    def test_quadratic_steps_case1():
      sampler = make_sampler({ 'steps skip type': 'quadratic' })
      steps, prevSteps = sampler._stepsSequence(21, 3, None)
      tf.assert_equal(steps, [20, 19, 11, 7, 5, 4])
      tf.assert_equal(prevSteps, [19, 11, 7, 5, 4, 3])
      return
    
    for K in range(1, 10):
      test_common({ 'steps skip type': { 'kind': 'uniform', 'K': K } })
    test_common({ 'steps skip type': 'quadratic' })

    test_uniform_steps()
    test_quadratic_steps_case1()
    test_quadratic_steps_no_duplicate()
    return
  
  def DDPMTests():
    DDPM_CONFIG = {
      'kind': 'ddpm',
      'noise stddev': 'zero',
    }
    def make_sampler(config):
      return sampler_from_config(dict(DDPM_CONFIG, **config))
    
    def test_steps():
      sampler = make_sampler({})
      steps = sampler._stepsSequence(10, 0, None)
      tf.assert_equal(steps, [9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
      return
    
    test_steps()
    return
  
  DDIMTests()
  DDPMTests()
  pass