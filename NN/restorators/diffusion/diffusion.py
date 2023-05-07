import tensorflow as tf
import numpy as np
from ..IRestorationProcess import IRestorationProcess
from .diffusion_schedulers import CDiffusionParameters, CDPDiscrete, get_beta_schedule
from .diffusion_samplers import sampler_from_config
from ..source_distributions import source_distribution_from_config

# simple gaussian diffusion process
class CGaussianDiffusion(IRestorationProcess):
  def __init__(self, 
    channels,
    schedule,
    sampler,
    sourceDistribution,
    lossScaling=None,
  ):
    super().__init__(channels)
    self._lossScaling = self._get_loss_scaling(lossScaling)
    self._schedule = schedule
    self._sampler = sampler
    self._sourceDistribution = sourceDistribution
    return
  
  def _forwardStep(self, x0, noise, t):
    (alphaHatT, SNR, ) = self._schedule.parametersForT(t, [CDiffusionParameters.PARAM_ALPHA_HAT, CDiffusionParameters.PARAM_SNR, ])
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
  
  def forward(self, x0):
    '''
    This function implements the forward diffusion process. It takes an initial value and applies the T steps of the diffusion process.
    '''
    # source distribution need to know the shape of the input, so we need to ensure it explicitly
    x0 = tf.ensure_shape(x0, (None, self._channels))
    sampled = self._sourceDistribution.sampleFor(x0)
    x1 = sampled['xT']
    T = sampled['T']
    # convert to discrete time, with floor rounding
    t = tf.cast(tf.floor(T * self._schedule.noise_steps), tf.int32)
    t = tf.clip_by_value(t, 0, self._schedule.noise_steps - 1)

    tf.assert_equal(tf.shape(x0), tf.shape(x1))
    tf.assert_equal(tf.shape(x0)[:1], tf.shape(t)[:1])
    return self._forwardStep(x0, x1, t)
  
  def _makeDenoiser(self, denoiser, modelT, valueShape):
    if self._schedule.is_discrete: # discrete time diffusion
      noise_steps = self._schedule.noise_steps
      encodedT = tf.cast(tf.range(noise_steps), tf.float32) / noise_steps
      encodedT = tf.reshape(encodedT, (noise_steps, 1))
      if not(modelT is None):
        encodedT = modelT(encodedT) # (noise_steps, M)
      M = tf.shape(encodedT)[-1]
      tf.assert_equal(tf.shape(encodedT), (noise_steps, M))

      def predictNoise(x, t):
        B = valueShape[0]
        # populate encoded T
        T = tf.gather(encodedT, t)
        T = tf.reshape(T, (1, M))
        T = tf.tile(T, (B, 1))
        tf.assert_equal(tf.shape(T), (B, M))
        tf.assert_equal(tf.shape(x), valueShape)
        return denoiser(x=x, t=T)
      
      return predictNoise
    
    raise NotImplementedError('Continuous time diffusion is not implemented yet')
  
  def reverse(self, value, denoiser, modelT=None, startStep=None, endStep=0, **kwargs):
    # NOTE: don't use 'early stopping' here, like in the autoregressive case
    sampler = kwargs.get('sampler', self._sampler)
    if isinstance(value, tuple):
      value = self._sourceDistribution.initialValueFor(value + (self._channels, ))
  
    initShape = tf.shape(value)
    denoiser = self._makeDenoiser(denoiser, modelT, initShape)
    value = sampler.sample(
      value,
      model=denoiser,
      schedule=self._schedule,
      # TODO: get startStep/endStep from the schedule
      startStep=startStep, endStep=endStep,
      **kwargs
    )
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
  name = config['kind'].lower()
  # TODO: add support for time schedule via source distribution
  # t_schedule = None
  # TShedule = config.get('T_schedule', None)
  # assert TShedule in [None, 'adjusted'], 'Unknown T_schedule'
  # if 'adjusted' == TShedule:
  #   t_schedule = adjustedTSampling

  if 'ddpm' == name:
    sourceDistributionConfigs = config['source distribution']
    return CGaussianDiffusion(
      channels=config['channels'],
      schedule=CDPDiscrete(
        beta_schedule=get_beta_schedule(config['beta_schedule']),
        noise_steps=config['noise_steps']
      ),
      sampler=sampler_from_config(config['sampler']),
      lossScaling=config['loss_scaling'],
      sourceDistribution=source_distribution_from_config(sourceDistributionConfigs)
    )

  raise ValueError('Unknown diffusion name')