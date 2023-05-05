import tensorflow as tf
import numpy as np

# schedulers
def cosine_beta_schedule(timesteps, s=0.008):
  # cosine schedule as proposed in https://arxiv.org/abs/2102.09672
  def f(t):
    t = t / tf.cast(timesteps, tf.float32)
    fraction = (t + s) / (1 + s)
    alphas = tf.cos(fraction * np.pi / 2) ** 2
    alpha0 = tf.cos((s / (1 + s)) * np.pi / 2) ** 2
    return alphas / alpha0

  f_t = f( tf.linspace(0.0, timesteps, timesteps + 1) )
  betas = 1.0 - (f_t[1:] / f_t[:-1])
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
  PARAM_BETA = 0
  PARAM_ALPHA = 1
  PARAM_ALPHA_HAT = 2
  PARAM_POSTERIOR_VARIANCE = 3
  PARAM_SNR = 4

  def parametersForT(self, T):
    raise NotImplementedError("parametersForT not implemented")
  

  @property
  def is_discrete(self):
    raise NotImplementedError("is_discrete not implemented")
  pass
  
class CDPDiscrete(CDiffusionParameters):
  def __init__(self, beta_schedule, noise_steps):
    super().__init__()
    self._noise_steps = noise_steps
    beta = beta_schedule
    if callable(beta_schedule):
      beta = beta_schedule(self.noise_steps)
    beta = tf.cast(beta, tf.float32)
    tf.assert_equal(tf.shape(beta), (self.noise_steps, ))

    alpha_hat = tf.math.cumprod(1. - beta)
    # due to this, posterior variance of step 0 is 0.0
    alpha_hat_prev = tf.concat([[1.0], alpha_hat[:-1]], axis=-1)
    
    # calculations for posterior q(x_{t-1} | x_t, x_0)
    # see https://calvinyluo.com/2022/08/26/diffusion-tutorial.html Eq. 54
    posterior_variance = beta * (1. - alpha_hat_prev) / (1. - alpha_hat)
    posterior_variance = tf.where(tf.math.is_finite(posterior_variance), posterior_variance, 0.0) # just in case
    # see https://calvinyluo.com/2022/08/26/diffusion-tutorial.html Eq. 64
    SNR = alpha_hat / (1. - alpha_hat)

    self._steps = tf.stack([beta, 1.0 - beta, alpha_hat, posterior_variance, SNR], axis=-1)
    return

  def parametersForT(self, T, index):
    p = tf.gather(self._steps, T)
    return[tf.reshape(p[..., i], (-1, 1)) for i in index]
  
  def debugParams(self):
    values = self._steps.numpy()
    for i, step in enumerate(values):
      print("{:d} | beta: {:.5f}, alpha: {:.5f}, alpha_hat: {:.5f}, posterior_variance: {:.5f}, SNR: {:.15f}".format(i, *step))
    return

  @property
  def noise_steps(self):
    return self._noise_steps
  
  @property
  def is_discrete(self): return True
# End of CDPDiscrete 
