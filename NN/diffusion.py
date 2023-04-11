import tensorflow as tf
import numpy as np
from NN.IRestorationProcess import IRestorationProcess

# shedulers
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
    
##########################
# simple gaussian diffusion process
class CGaussianDiffusion(IRestorationProcess):
  PARAM_BETA = 0
  PARAM_ALPHA = 1
  PARAM_ALPHA_HAT = 2
  PARAM_POSTERIOR_VARIANCE = 3
  PARAM_SNR = 4

  def __init__(self, 
    channels,
    beta_shedule,
    noise_steps,
    t_shedule=None,
    noise_fn=None,
    lossScaling=None,
  ):
    super().__init__(channels)
    self._lossScaling = self._get_loss_scaling(lossScaling)
    self._tshedule = t_shedule
    self._noise_fn = noise_fn
    self.noise_steps = noise_steps
    
    beta = beta_shedule(self.noise_steps) if callable(beta_shedule) else beta_shedule
    beta = tf.cast(beta, tf.float32)
    tf.assert_equal(tf.shape(beta), (self.noise_steps, ))

    alpha = 1. - beta
    alpha_hat = tf.math.cumprod(alpha)
    # due to this, posterior variance of step 0 is 0.0
    alpha_hat_prev = tf.concat([[1.0], alpha_hat[:-1]], axis=-1)
    
    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = beta * (1. - alpha_hat_prev) / (1. - alpha_hat)

    # see https://calvinyluo.com/2022/08/26/diffusion-tutorial.html#mjx-eqn%3Aeq%3A102
    SNR = alpha_hat / (1. - alpha_hat)

    self._steps = tf.stack([beta, alpha, alpha_hat, posterior_variance, SNR], axis=-1)
    return

  def _TParams(self, T, index):
    p = tf.gather(self._steps, T)
    return[tf.reshape(p[..., i], (-1, 1)) for i in index]
  
  def debugParams(self):
    values = self._steps.numpy()
    for i, step in enumerate(values):
      print("{:d} | beta: {:.5f}, alpha: {:.5f}, alpha_hat: {:.5f}, posterior_variance: {:.5f}, SNR: {:.15f}".format(i, *step))
    return

  def _sampleT(self, TShape):
    TShape = (TShape[0], 1)
    if self._tshedule is not None:
      return self._tshedule(self.noise_steps, TShape)
    
    return tf.random.uniform(minval=0, maxval=self.noise_steps, shape=TShape, dtype=tf.int32)
    
  def _noise(self, s):
    if self._noise_fn is not None:
      return self._noise_fn(s)
    return tf.random.normal(s)

  def forward(self, x0, t=None):
    '''
    This function implements the forward diffusion process. It takes an initial value and applies the T steps of the diffusion process.
    '''
    s = tf.shape(x0)
    if t is None:
      t = self._sampleT(s[:-1])
      
    (alphaHatT, SNR, ) = self._TParams(t, [self.PARAM_ALPHA_HAT, self.PARAM_SNR, ])
    noise = self._noise(s)
    xT = (tf.sqrt(alphaHatT) * x0) + (tf.sqrt(1.0 - alphaHatT) * noise)
    tf.assert_equal(tf.shape(xT), tf.shape(x0))
    return {
      'xT': xT,
      't': t, # discrete time
      'T': tf.cast(t, tf.float32) / self.noise_steps, # continuous time
      'target': noise,
      'SNR': SNR,
    }
  
  def reverse(self, value, denoiser, modelT=None, startStep=None, endStep=0):
    # NOTE: don't use 'early stopping' here, like in the autoregressive case
    if isinstance(value, tuple):
      value = tf.random.normal(value + (self._channels, ), dtype=tf.float32)
  
    encodedT = tf.cast(tf.range(self.noise_steps), value.dtype) / self.noise_steps
    encodedT = tf.reshape(encodedT, (self.noise_steps, 1))
    if not(modelT is None):
      encodedT = modelT(encodedT) # (self.noise_steps, M)
    M = tf.shape(encodedT)[-1]
    tf.assert_equal(tf.shape(encodedT), (self.noise_steps, M))
    initShape = tf.shape(value)
    B = initShape[0]

    def predictNoise(x, t):
      # populate encoded T
      T = tf.gather(encodedT, t)
      T = tf.reshape(T, (1, M))
      T = tf.tile(T, (B, 1))
      tf.assert_equal(tf.shape(T), (B, M))
      tf.assert_equal(tf.shape(x), initShape)
      return denoiser(x=x, t=T)
    #######################
    def reverseStep(x, t):
      predictedNoise = predictNoise(x, t)
      # obtain parameters for the current step
      (variance, alpha, beta, alpha_hat) = self._TParams(t, [
        self.PARAM_POSTERIOR_VARIANCE,
        self.PARAM_ALPHA,
        self.PARAM_BETA,
        self.PARAM_ALPHA_HAT,
      ])
      # scale predicted noise
      # s = (1 - alpha) / sqrt(1 - alpha_hat)) = beta / sqrt(1 - alpha_hat)
      s = beta / tf.sqrt(1.0 - alpha_hat)
      x_prev = (x - s * predictedNoise) / tf.sqrt(alpha)
      tf.assert_equal(tf.shape(x_prev), initShape)
      return(x_prev, variance)
    #######################
    if startStep is None:
      startStep = self.noise_steps - 1
    step = tf.constant(startStep, dtype=tf.int32)
    while endStep <= step:
      value, stddev = reverseStep(value, step)
      value += self._noise(tf.shape(value)) * stddev
      tf.assert_equal(tf.shape(value), initShape)
      step -= 1
      continue
    tf.assert_equal(tf.shape(value), initShape)
    return value
  
  def calculate_loss(self, x_hat, predicted):
    target = x_hat['target']
    tf.assert_equal(tf.shape(target), tf.shape(predicted))
    B = tf.shape(predicted)[0]
    
    loss = tf.losses.mae(target, predicted)
    tf.assert_equal(tf.shape(loss), (B, ))

    loss = self._lossScaling(loss, x_hat, predicted)
    return loss
  
  def _get_loss_scaling(self, args):
    if args is None:
      return lambda loss, x_hat, predicted: loss
    
    name = args['name']
    gamma = args.get('gamma', None)
    if 'min SNR' == name:
      return lambda loss, x_hat, predicted: loss * tf.nn.minimum(gamma, x_hat['SNR'])
  
    if 'max SNR' == name:
      return lambda loss, x_hat, predicted: loss * tf.nn.maximum(gamma, x_hat['SNR'])
 
    if 'clip SNR' == name:
      minSNR = args['min']
      maxSNR = args['max']
      return lambda loss, x_hat, predicted: loss * tf.clip_by_value(x_hat['SNR'], minSNR, maxSNR)
    
    raise ValueError('Unknown loss scaling')
  
##########################
def adjustedTSampling(noise_steps, TShape):
  # oversample the first few steps, as they are more important
  index = tf.linspace(0., 1., 1 + noise_steps)[1:]
  mu = -.1
  sigma = 1.85
  pdfA = tf.exp(-tf.square(tf.math.log(index) - mu) / (2. * sigma * sigma))
  pdfB = index * (sigma * np.sqrt(2. * np.pi))
  pdf = pdfA / pdfB

  weights = tf.nn.softmax(pdf, axis=-1)[None]
  res = tf.random.categorical(tf.math.log(weights), tf.math.reduce_prod(TShape))
  return tf.reshape(res, )

def diffusion_from_config(config):
  t_shedule = None
  TShedule = config.get('T_shedule', None)
  assert TShedule in [None, 'adjusted'], 'Unknown T_shedule'
  if 'adjusted' == TShedule:
    t_shedule = adjustedTSampling

  kind = config['kind'].lower()
  assert kind in ['ddpm'], 'Unknown diffusion kind'
  if 'ddpm' == kind:
    return CGaussianDiffusion(
      channels=config['channels'],
      beta_shedule=get_beta_schedule(config['beta_schedule']),
      noise_steps=config['noise_steps'],
      lossScaling=config['loss_scaling'],
      t_shedule=t_shedule
    )

  raise ValueError('Unknown diffusion kind')

if '__main__' == __name__:
  from Utils.utils import setupGPU
  setupGPU()
  d = CGaussianDiffusion(channels=3, beta_shedule=get_beta_schedule('cosine'), noise_steps=100)
  d.debugParams()