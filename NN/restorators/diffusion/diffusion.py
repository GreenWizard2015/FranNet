import tensorflow as tf
import numpy as np
from ..IRestorationProcess import IRestorationProcess
from .diffusion_schedulers import schedule_from_config
from .diffusion_samplers import sampler_from_config
from ..source_distributions import source_distribution_from_config
from ..timeHelper import make_discrete_time_encoder

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
  
  def _extractTime(self, kwargs):
    if 'T' in kwargs:
      return self._schedule.to_discrete(kwargs['T']), kwargs['T']
    
    return self._schedule.to_discrete(kwargs['t']), kwargs['t']
  
  def _forwardStep(self, x0, noise, **kwargs):
    t, T = self._extractTime(kwargs)
    tf.assert_equal(tf.shape(t)[1], 1)
    tf.assert_equal(tf.shape(t), tf.shape(T))
    tf.assert_equal(tf.rank(t), 2)
    step = self._schedule.parametersForT(t)
    alphaHatT = step.alphaHat

    signal_rate, noise_rate = tf.sqrt(alphaHatT), tf.sqrt(1.0 - alphaHatT)
    xT = (signal_rate * x0) + (noise_rate * noise)
    tf.assert_equal(tf.shape(xT), tf.shape(x0))
    tf.assert_equal(tf.shape(x0)[:1], tf.shape(t)[:1])
    return {
      'xT': xT,
      't': t, # discrete time
      'T': T, # continuous time
      'target': noise,
      'SNR': step.SNR,
    }
  
  def forward(self, x0):
    '''
    This function implements the forward diffusion process. It takes an initial value and applies the T steps of the diffusion process.
    '''
    # source distribution need to know the shape of the input, so we need to ensure it explicitly
    x0 = tf.ensure_shape(x0, (None, self._channels))
    sampled = self._sourceDistribution.sampleFor(x0)
    x1 = sampled['xT']
    tf.assert_equal(tf.shape(x0), tf.shape(x1))
    return self._forwardStep(x0, x1, T=sampled['T'])
  
  def _makeDenoiser(self, denoiser, modelT, valueShape):
    if self._schedule.is_discrete: # discrete time diffusion
      noise_steps = self._schedule.noise_steps
      
      timeEncoder = make_discrete_time_encoder(
        modelT=modelT,
        allT=self._schedule.to_continuous( tf.range(noise_steps)[:, None] )
      )

      def predictNoise(x, T, **kwargs):
        B = valueShape[0]
        # populate encoded T
        T = timeEncoder(t=T, B=B)
        tf.assert_equal(tf.shape(T)[:1], (B,))
        tf.assert_equal(tf.shape(x), valueShape)
        return denoiser(x=x, t=T)
      
      return predictNoise
    
    raise NotImplementedError('Continuous time diffusion is not implemented yet')
  
  def reverse(self, value, denoiser, modelT=None, **kwargs):
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
      schedule=schedule_from_config(config['schedule']),
      sampler=sampler_from_config(config['sampler']),
      lossScaling=config['loss scaling'],
      sourceDistribution=source_distribution_from_config(sourceDistributionConfigs)
    )

  raise ValueError('Unknown diffusion name')