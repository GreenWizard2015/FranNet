import tensorflow as tf
from .IRestorationProcess import IRestorationProcess
from .source_distributions import source_distribution_from_config
from .timeHelper import make_time_encoder

'''
Autoregressive restoration process
'''
class CARProcess(IRestorationProcess):
  def __init__(self, channels, sourceDistribution, sampler):
    super().__init__(channels)
    self._sourceDistribution = sourceDistribution
    self._sampler = sampler
    return
  
  def forward(self, x0):
    B = tf.shape(x0)[0]
    # source distribution need to know the shape of the input, so we need to ensure it explicitly
    x0 = tf.ensure_shape(x0, (None, self._channels))
    sampled = self._sourceDistribution.sampleFor(x0)
    xT = sampled['xT']
    
    tf.assert_equal(tf.shape(x0), (B, self._channels))
    tf.assert_equal(tf.shape(x0), tf.shape(xT))
    return self._sampler.train(x0=x0, x1=xT, T=sampled['T'])
  
  def calculate_loss(self, x_hat, predicted, **kwargs):
    lossFn = kwargs.get('lossFn', tf.losses.mae) # default loss function
    return lossFn(x_hat['target'], predicted)

  def _makeDenoiser(self, model, modelT):
    timeEncoder = make_time_encoder(modelT)

    def denoiser(x, t=None, **kwargs):
      B = tf.shape(x)[0]
      T = timeEncoder(t=t, B=B)
      tf.assert_equal(B, tf.shape(T)[0])
      return model(x=x, t=T, mask=kwargs.get('mask', None))
    return denoiser

  def reverse(self, value, denoiser, modelT=None, **kwargs):
    if isinstance(value, tuple):
      value = self._sourceDistribution.initialValueFor(value + (self._channels, ))

    denoiser = self._makeDenoiser(denoiser, modelT)
    res = self._sampler.sample(value=value, model=denoiser, **kwargs)
    tf.assert_equal(tf.shape(res), tf.shape(value))
    return res
# End of CARProcess

def autoregressive_restoration_from_config(config):
  assert 'autoregressive' == config['name']
  from .samplers import sampler_from_config
  
  return CARProcess(
    channels=config['channels'],
    sourceDistribution=source_distribution_from_config(config['source distribution']),
    sampler=sampler_from_config(config['sampler'])
  )