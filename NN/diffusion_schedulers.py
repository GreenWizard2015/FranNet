import tensorflow as tf
import numpy as np

# schedulers
def cosine_beta_schedule(timesteps, s=0.008):
  """
  cosine schedule as proposed in https://arxiv.org/abs/2102.09672
  """
  x = tf.linspace(0.0, timesteps, timesteps + 1)
  alphas_cumprod = tf.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
  alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
  betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
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
  
  def sampleT(self, TShape):
    raise NotImplementedError("sampleT not implemented")
  pass
  
class CDPDiscrete(CDiffusionParameters):
  def __init__(self, beta_schedule, noise_steps, t_schedule=None):
    super().__init__()
    self._tschedule = t_schedule
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
 
  def sampleT(self, TShape):
    TShape = (TShape[0], 1)
    if self._tschedule is not None:
      return self._tschedule(self.noise_steps, TShape)
    
    return tf.random.uniform(minval=0, maxval=self.noise_steps, shape=TShape, dtype=tf.int32)
     
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
