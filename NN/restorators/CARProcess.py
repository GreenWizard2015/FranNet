import tensorflow as tf
from .IRestorationProcess import IRestorationProcess
from .source_distributions import source_distribution_from_config

'''
Simplest autoregressive restoration process:
Predicts the derivative/direction from given values towards the ground truth values.
This approach doesn't use the time parameter; it always zeroes it out.
Theoretically, it should be able to work with arbitrary distributions, not only N(0.0, 1.0).
'''
class CARProcess(IRestorationProcess):
  def __init__(self, 
    channels,
    sourceDistribution,
    sampling,
  ):
    super().__init__(channels)
    self._sampling = sampling
    self._sourceDistribution = sourceDistribution
    return
  
  def forward(self, x0):
    B = tf.shape(x0)[0]
    # source distribution need to know the shape of the input, so we need to ensure it explicitly
    x0 = tf.ensure_shape(x0, (None, self._channels))
    sampled = self._sourceDistribution.sampleFor(x0)
    xT = sampled['xT']
    T = tf.zeros((B, 1), dtype=tf.float32)
    
    tf.assert_equal(tf.shape(x0), (B, self._channels))
    tf.assert_equal(tf.shape(x0), tf.shape(xT))
    return { 'x0': x0, 'xT': xT, 'T': T }
  
  def calculate_loss(self, x_hat, predicted):
    delta = x_hat['x0'] - x_hat['xT']
    tf.assert_equal(tf.shape(delta), tf.shape(predicted))
    return tf.losses.mae(delta, predicted)
  
  # TODO: move this to a separate class
  @tf.function
  def _dynamicAR(self, values, model, sampling=None):
    sampling = sampling or {}
    sampling = {**self._sampling, **sampling}
    ##################
    B = tf.shape(values)[0]
    decay = sampling['step size decay']
    threshold = sampling['threshold']
    stepsLimit = sampling['steps limit']

    allIndices = tf.range(B)[..., None]
    msk = tf.fill((B,), True)
    ittr = tf.constant(0)
    stepSize = 1.0
    while tf.logical_and(tf.reduce_any(msk), ittr < stepsLimit):
      nextStepSize = stepSize * decay
      # add noise to copy of the input values
      noise = tf.random.normal(tf.shape(values), stddev=tf.square(nextStepSize))
      V_hat = values + noise
      dxdt = model(x=V_hat, mask=msk)
      newValues = tf.boolean_mask(V_hat, msk, axis=0) + dxdt * stepSize
      # update only those values that were changed
      indices = tf.boolean_mask(allIndices, msk, axis=0)
      newValues = tf.tensor_scatter_nd_update(values, indices, newValues)
      # if "pixel" was changed by less than threshold, then it is considered converged
      msk = threshold < tf.reduce_max(tf.abs(newValues - values), axis=-1)
      tf.assert_equal(tf.shape(msk), tf.shape(values)[:-1])

      stepSize = nextStepSize
      values = newValues
      ittr += 1
      continue

    return values

  def _makeDenoiser(self, model, modelT, shp):
    T = [[0.0]]
    if not(modelT is None):
      T = modelT(T)[0]
    #######################
    T = tf.reshape(T, (1, ) * len(shp) + (-1, ))
    T = tf.tile(T, tf.concat([shp, [1]], axis=0))

    def denoiser(x, mask):
      tf.assert_equal(tf.shape(x)[0], tf.shape(T)[0])
      return model(x=x, t=T, mask=mask)
    return denoiser

  def reverse(self, value, denoiser, modelT=None, sampling=None):
    if isinstance(value, tuple):
      value = self._sourceDistribution.initialValueFor(value + (self._channels, ))

    denoiser = self._makeDenoiser(denoiser, modelT, tf.shape(value)[:-1])
    res = self._dynamicAR(values=value, model=denoiser, sampling=sampling)
    tf.assert_equal(tf.shape(res), tf.shape(value))
    return res
# End of CARProcess

def autoregressive_restoration_from_config(config):
  assert 'autoregressive' == config['name']

  sampling = config['sampling']
  sampling = {
    'threshold': sampling['threshold'],
    'steps limit': sampling['steps limit'],
    'step size decay': sampling['step size decay'],
  }

  channels = config['channels']
  sourceDistributionConfigs = config['source distribution']
  return CARProcess(
    channels=channels,
    sourceDistribution=source_distribution_from_config(sourceDistributionConfigs),
    sampling=sampling,
  )