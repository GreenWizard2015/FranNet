import tensorflow as tf
import numpy as np
from NN.IRestorationProcess import IRestorationProcess
from NN.diffusion_schedulers import CDiffusionParameters, CDPDiscrete, get_beta_schedule

# simple gaussian diffusion process
class CGaussianDiffusion(IRestorationProcess):
  def __init__(self, 
    channels,
    schedule,
    noise_fn=None,
    lossScaling=None,
  ):
    super().__init__(channels)
    self._lossScaling = self._get_loss_scaling(lossScaling)
    self._noise_fn = noise_fn
    self._schedule = schedule
    return
  
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
      t = self._schedule.sampleT(s[:-1])

    (alphaHatT, SNR, ) = self._schedule.parametersForT(t, [CDiffusionParameters.PARAM_ALPHA_HAT, CDiffusionParameters.PARAM_SNR, ])
    noise = self._noise(s)
    signal_rate, noise_rate = tf.sqrt(alphaHatT), tf.sqrt(1.0 - alphaHatT)
    xT = (signal_rate * x0) + (noise_rate * noise)
    tf.assert_equal(tf.shape(xT), tf.shape(x0))
    return {
      'xT': xT,
      't': t, # discrete time
      'T': tf.cast(t, tf.float32) / self._schedule.noise_steps, # continuous time
      'target': noise,
      'SNR': SNR,
    }
  
  def reverse(self, value, denoiser, modelT=None, startStep=None, endStep=0):
    # NOTE: don't use 'early stopping' here, like in the autoregressive case
    if isinstance(value, tuple):
      value = tf.random.normal(value + (self._channels, ), dtype=tf.float32)
  
    noise_steps = self._schedule.noise_steps
    encodedT = tf.cast(tf.range(noise_steps), value.dtype) / noise_steps
    encodedT = tf.reshape(encodedT, (noise_steps, 1))
    if not(modelT is None):
      encodedT = modelT(encodedT) # (noise_steps, M)
    M = tf.shape(encodedT)[-1]
    tf.assert_equal(tf.shape(encodedT), (noise_steps, M))
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
      (variance, alpha, beta, alpha_hat) = self._schedule.parametersForT(t, [
        CDiffusionParameters.PARAM_POSTERIOR_VARIANCE,
        CDiffusionParameters.PARAM_ALPHA,
        CDiffusionParameters.PARAM_BETA,
        CDiffusionParameters.PARAM_ALPHA_HAT,
      ])
      # scale predicted noise
      # s = (1 - alpha) / sqrt(1 - alpha_hat)) = beta / sqrt(1 - alpha_hat)
      s = beta / tf.sqrt(1.0 - alpha_hat)
      x_prev = (x - s * predictedNoise) / tf.sqrt(alpha)
      tf.assert_equal(tf.shape(x_prev), initShape)
      return(x_prev, variance)
    #######################
    if startStep is None:
      startStep = noise_steps - 1
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
  t_schedule = None
  TShedule = config.get('T_schedule', None)
  assert TShedule in [None, 'adjusted'], 'Unknown T_schedule'
  if 'adjusted' == TShedule:
    t_schedule = adjustedTSampling

  kind = config['kind'].lower()
  assert kind in ['ddpm'], 'Unknown diffusion kind'
  if 'ddpm' == kind:
    return CGaussianDiffusion(
      channels=config['channels'],
      schedule=CDPDiscrete(
        beta_schedule=get_beta_schedule(config['beta_schedule']),
        noise_steps=config['noise_steps'],
        t_schedule=t_schedule
      ),
      lossScaling=config['loss_scaling']
    )

  raise ValueError('Unknown diffusion kind')