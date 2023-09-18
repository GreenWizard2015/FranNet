import tensorflow as tf
import numpy as np
from NN.utils import make_steps_sequence
from Utils.utils import CFakeObject

# schedulers
def cosine_beta_schedule(timesteps, S=0.008):
  # cosine schedule as proposed in https://arxiv.org/abs/2102.09672
  def f(t):
    t = tf.cast(t, tf.float64)
    s = tf.cast(S, t.dtype)
    pi = tf.cast(np.pi, t.dtype)

    t = t / tf.cast(timesteps, t.dtype)
    fraction = (t + s) / (1 + s)
    alphas = tf.cos(fraction * pi / 2) ** 2
    alpha0 = tf.cos((s / (1 + s)) * pi / 2) ** 2
    return alphas / alpha0

  f_t = f( tf.linspace(0.0, timesteps, num=timesteps + 1) )
  betas = 1.0 - (f_t[1:] / f_t[:-1])
  # NOTE: default value 0.9999 is used in the official implementation, but may cause numerical issues during sampling
  #       if not clip sampled values
  #       alternatively, we can change 0.9999 to 0.99
  return tf.clip_by_value(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
  return tf.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
  return tf.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
  betas = tf.linspace(-6, 6, timesteps)
  return tf.nn.sigmoid(betas) * (beta_end - beta_start) + beta_start

def get_beta_schedule(name):
  if name == "cosine":
    return cosine_beta_schedule
  elif name == "linear":
    return linear_beta_schedule
  elif name == "quadratic":
    return quadratic_beta_schedule
  elif name == "sigmoid":
    return sigmoid_beta_schedule
  raise ValueError("Unknown beta schedule name: {}".format(name))

class CDiffusionParameters:
  def parametersForT(self, T):
    raise NotImplementedError("parametersForT not implemented")
  
  @property
  def is_discrete(self):
    raise NotImplementedError("is_discrete not implemented")
  
  def to_continuous(self, t):
    raise NotImplementedError("to_continuous not implemented")
  
  def to_discrete(self, t, lastStep=False):
    raise NotImplementedError("to_discrete not implemented")
  
  # helper function to calculate posterior variance between two specified steps
  def varianceBetween(self, alpha_hat_t, alpha_hat_t_prev):
    beta_hat_t = 1.0 - alpha_hat_t
    beta_hat_t_prev = 1.0 - alpha_hat_t_prev
    variance = (beta_hat_t_prev / beta_hat_t) * (1.0 - alpha_hat_t / alpha_hat_t_prev)
    return variance
# End of CDiffusionParameters
  
class CDPDiscrete(CDiffusionParameters):
  def __init__(self, beta_schedule, noise_steps):
    super().__init__()
    self._noise_steps = noise_steps
    beta = beta_schedule
    if callable(beta_schedule):
      beta = beta_schedule(self.noise_steps)
    beta = tf.cast(beta, tf.float64)
    tf.assert_equal(tf.shape(beta), (self.noise_steps, ))

    parameters = {
      'beta': beta,
      'alpha': 1.0 - beta,
    }

    parameters['alphaHat'] = alpha_hat = tf.math.cumprod(1. - beta)
    # due to this, posterior variance of step 0 is 0.0
    alpha_hat_prev = tf.concat([[1.0], alpha_hat[:-1]], axis=-1)
    
    # calculations for posterior q(x_{t-1} | x_t, x_0)
    # see https://calvinyluo.com/2022/08/26/diffusion-tutorial.html Eq. 54
    posterior_variance = beta * (1. - alpha_hat_prev) / (1. - alpha_hat)
    posterior_variance = tf.where(tf.math.is_finite(posterior_variance), posterior_variance, 0.0) # just in case
    parameters['posteriorVariance'] = posterior_variance
    # see https://calvinyluo.com/2022/08/26/diffusion-tutorial.html Eq. 64
    parameters['SNR'] = alpha_hat / (1.0 - alpha_hat)

    # prepend "clean" step
    parameters['beta'] = tf.concat([[0.0], parameters['beta']], axis=-1)
    parameters['alpha'] = tf.concat([[1.0], parameters['alpha']], axis=-1)
    parameters['alphaHat'] = tf.concat([[1.0], parameters['alphaHat']], axis=-1)
    parameters['posteriorVariance'] = tf.concat([[0.0], parameters['posteriorVariance']], axis=-1)
    parameters['SNR'] = tf.concat([[float('inf')], parameters['SNR']], axis=-1)
    
    # some useful parameters
    parameters['sqrt_alpha'] = tf.sqrt(parameters['alpha'])
    parameters['sqrt_one_minus_alpha'] = tf.sqrt(parameters['beta'])

    parameters['sqrt_alpha_hat'] = tf.sqrt(parameters['alphaHat'])
    parameters['sqrt_one_minus_alpha_hat'] = tf.sqrt(1.0 - parameters['alphaHat'])

    parameters['one_minus_alpha'] = parameters['beta']
    parameters['one_minus_alpha_hat'] = 1.0 - parameters['alphaHat']
    
    parameters['sigma'] = tf.sqrt(parameters['posteriorVariance'])

    self._steps = parameters
    return

  def parametersForT(self, T, dtype=tf.float32):
    tf.debugging.assert_less(T, self.noise_steps, "T must be less than noise_steps")

    T = tf.cast(T, tf.int32) + 1 # shifted by 1
    tf.debugging.assert_greater_equal(T, 0, "T must be non-negative")

    # Collect all parameters
    def F(x):
      x = tf.gather(x, T)
      x = tf.reshape(x, tf.shape(T))
      return tf.cast(x, dtype)
    
    res = {k: F(v) for k, v in self._steps.items()}
    return CFakeObject(**res)

  @property
  def noise_steps(self):
    return self._noise_steps
  
  @property
  def is_discrete(self): return True

  def to_continuous(self, t):
    N = self.noise_steps - 1
    tf.assert_equal(t.dtype, tf.int32)
    tf.debugging.assert_greater_equal(t, 0)
    tf.debugging.assert_less_equal(t, N)
    t = tf.cast(t, tf.float32)
    return tf.clip_by_value(t / float(N), 0.0, 1.0)
  
  def to_discrete(self, t, lastStep=False):
    tf.assert_equal(t.dtype, tf.float32)
    tf.debugging.assert_less_equal(0.0, t)
    tf.debugging.assert_less_equal(t, 1.0)
    
    # convert to discrete time, with floor rounding
    N = self.noise_steps if lastStep else self.noise_steps - 1
    res = tf.cast(tf.floor(t * N), tf.int32)
    # clip to [0, noise_steps - 1] even if t == 1.0
    return tf.clip_by_value(res, 0, self.noise_steps - 1)
  
  # helper function to create a sequence of steps
  def steps_sequence(self, startStep, endStep, config, reverse=False):
    maxSteps = self.noise_steps
    if startStep is None: startStep = maxSteps
    if endStep is None: endStep = 0
    
    # check bounds
    tf.debugging.assert_greater_equal(startStep, 0)
    tf.debugging.assert_less_equal(startStep, maxSteps)
    tf.debugging.assert_greater_equal(endStep, 0)
    tf.debugging.assert_less_equal(endStep, maxSteps)
    
    res = make_steps_sequence( startStep=startStep, endStep=endStep - 1, config=config )
    if reverse: res = tuple(x[::-1] for x in res)
    return res
# End of CDPDiscrete 

def schedule_from_config(config):
  name = config['name'].lower()
  if name == "discrete":
    return CDPDiscrete(
      beta_schedule=get_beta_schedule(config['beta schedule']),
      noise_steps=config['timesteps']
    )
  
  raise ValueError("Unknown beta schedule name: {}".format(name))