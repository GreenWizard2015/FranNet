import tensorflow as tf
from .IRestorationProcess import IRestorationProcess
import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions

def normVec(x):
  V, L = tf.linalg.normalize(x, axis=-1)
  V = tf.where(tf.math.is_nan(V), 0.0, V)
  return(V, L)

def normal_source_distribution(N):
  return tf.random.normal((N, 3), dtype=tf.float32)

def shaped_source_distribution(config):
  raise NotImplementedError('Not implemented yet')
  def _sample(N):
    direction = tfd.Uniform(low=[-1.0, -1.0, -1.0], high=[1.0, 1.0, 1.0])
    directions = direction.sample(N)
    directions, _ = normVec(directions)

    length = tfd.Normal(loc=0.0, scale=1.0)
    lengths = length.sample(N) ** 2.
    
    lengths = 5.0 + lengths
    return directions * lengths[..., None]
  
  return _sample

###################
class CDerivativeEncoder:
  def __init__(self, levels, scaleByCoef, scaleByTime):
    assert all([l != 0.5 for l in levels]), '0.5 is not invertible'
    
    self._scaleByCoef = scaleByCoef
    self._scaleByTime = scaleByTime
    self._forwardTransformation = list(levels)
    return
  
  def _encode(self, x0, xT):
    B, C = tf.shape(x0)[0], tf.shape(x0)[1]
    tf.assert_equal(tf.shape(xT), tf.shape(x0))
    tf.assert_equal(tf.shape(x0), (B, C))

    def derivative(x): return 2.0 * (x0 - x)
    K = []
    dxdt = derivative(xT)
    for coef in self._forwardTransformation:
      pt = xT + coef * dxdt
      dtdxAtPt = derivative(pt)

      # decrease the derivative amplitude by the coefficient, to make the process more stable
      if self._scaleByCoef and not (coef == 0.0):
        dtdxAtPt = dtdxAtPt / coef

      K.append(dtdxAtPt)
      continue

    K = tf.stack(K, axis=1)
    tf.assert_equal(tf.shape(K), (B, len(self._forwardTransformation), C))
    return K, dxdt
  
  def calculate_loss(self, x0, xT, T, predicted):
    K, dxdt = self._encode(x0, xT)
    tf.assert_equal(tf.shape(K), tf.shape(predicted))

    lossF = lambda a, b: tf.reduce_mean(tf.losses.mae(a, b))
    losses = []
    # main loss
    decoded = self.decode(predicted, T, training=True)
    tf.assert_equal(tf.shape(decoded), tf.shape(xT))
    losses.append( lossF(dxdt, decoded) )

    # additional supervision
    if self.useTime:
      predicted = predicted * T[..., None]
    # by each derivative level
    tf.assert_equal(tf.shape(K), tf.shape(predicted))
    losses.append( lossF(K, predicted) )

    return sum(losses, 0.0)
  
  def _inverseTransformation(self):
    coefs = tf.constant(self._forwardTransformation, dtype=tf.float32)
    res = 1.0 / (1.0 - 2.0 * coefs)
    if self._scaleByCoef: res = res * coefs
    res = res / tf.cast(tf.size(res), tf.float32) # mean
    tf.assert_equal(tf.shape(res), (len(self._forwardTransformation),))
    return res
  
  def decode(self, KPred, stepSize, training=False):
    coefficients = self._inverseTransformation()[None]
    N = tf.size(coefficients)
    B = tf.shape(KPred)[0]
    KPred = tf.reshape(KPred, (B, N, -1))
    C = tf.shape(KPred)[2]

    if self._scaleByTime:
      coefficients = coefficients * stepSize
    
    if not training:
      coefficients = coefficients * stepSize # normal step scaling

    res = KPred * coefficients[..., None]
    tf.assert_equal(tf.shape(res), (B, N, C))
    res = tf.reduce_sum(res, axis=1)
    tf.assert_equal(tf.shape(res), (B, C))
    return res
  
  @property
  def useTime(self):
    return self._scaleByTime
  
  @property
  def levels(self):
    return len(self._forwardTransformation)
###################
class CARProcess(IRestorationProcess):
  def __init__(self, 
    channels,
    extraTrainSteps,
    sourceDistributionSampler,
    sampling,
    derivativeEncoder,
  ):
    super().__init__(channels)
    self._sampling = sampling
    self._extraTrainSteps = extraTrainSteps
    self._sourceDistributionSampler = sourceDistributionSampler
    self._derivativeEncoder = derivativeEncoder
    return
    
  def _sampleT(self, x0, xT):
    B = tf.shape(x0)[0]
    if not self._derivativeEncoder.useTime:
      return tf.ones((B, 1), dtype=tf.float32)

    # some magic to make the T more suitable for the process. 
    # More focused on the small scale when we are close to the source x0
    T = tf.random.normal((B, 1), dtype=tf.float32)
    T = tf.square(T)
    _, L = normVec(xT - x0)
    T = T * L # time is proportional to the distance from the source
   
    return tf.clip_by_value(T, clip_value_min=1e-8, clip_value_max=1.0)

  def _mixValues(self, x0, x1):
    B = tf.shape(x0)[0]
    
    fraction = tf.random.uniform((B, 1))
    fraction = tf.pow(fraction, 2.0) # make it more likely to be close to 0, as it is more hard to predict
    fraction = tf.clip_by_value(fraction, clip_value_min=1e-8, clip_value_max=1.0 - 1e-8)
    # linear mix is less suitable for this process
    xT = (x0 * tf.sqrt(1 - fraction)) + (x1 * tf.sqrt(fraction))
    return xT
  
  def forward(self, x0, model=None):
    B = tf.shape(x0)[0]
    N = tf.shape(x0)[-1]
    xT = self._mixValues(
      x0=x0,
      x1=self._sourceDistributionSampler(B)
    )    

    if not(model is None):
      stepSize = tf.random.normal((B, 1), mean=0.0, stddev=1.0)
      # apply the update in the random direction
      xTNew = self._applyUpdate(xT, model(V=xT, T=stepSize), stepSize)
      # if the update is not valid, keep the old value
      xT = tf.where(tf.math.is_finite(xTNew), xTNew, xT)
      xT = tf.stop_gradient(xT)

    T = self._sampleT(x0, xT)
    tf.assert_equal(tf.shape(x0), (B, N))
    tf.assert_equal(tf.shape(x0), tf.shape(xT))
    tf.assert_equal(tf.shape(T), (B, 1))
    return { 'x0': x0, 'xT': xT, 'T': T }
  
  def train_step(self, x0, model):
    losses = []
    losses.append(super().train_step(x0, model)) # standard loss

    for _ in range(self._extraTrainSteps):
      x_hat = self.forward(x0, model=model)
      values = model(T=x_hat['T'], V=x_hat['xT'])
      losses.append(self.calculate_loss(x_hat, values))
      continue
    return sum(losses) / len(losses)
  
  @tf.function
  def _dynamicAR(self, values, model, sampling=None):
    if sampling is None:
      sampling = self._sampling
    else:
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
      newValues = self._applyUpdate(
        tf.boolean_mask(values, msk, axis=0),
        model(x=values, mask=msk, stepSize=stepSize),
        stepSize=stepSize,
        dynamic=sampling['dynamic adjustment']
      )
      stepSize = stepSize * decay
      
      indices = tf.boolean_mask(allIndices, msk, axis=0)
      newValues = tf.tensor_scatter_nd_update(values, indices, newValues)
      # if "pixel" was changed by less than threshold, then it is considered converged
      msk = threshold < tf.reduce_max(tf.abs(newValues - values), axis=-1)
      tf.assert_equal(tf.shape(msk), tf.shape(values)[:-1])

      # add noise to the active values
      noise = tf.random.normal(tf.shape(newValues), stddev=tf.square(stepSize))
      noise = tf.where(msk[..., None], noise, 0.0)
      tf.assert_equal(tf.shape(noise), tf.shape(newValues))
      values = newValues + noise
      ittr += 1
      continue

    return values

  def _makeDenoiser(self, model, modelT, shp):
    if self._derivativeEncoder.useTime:
      def denoiser(x, mask, stepSize):
        B = tf.shape(x)[0]
        # populate encoded T, use stepSize as time
        T = modelT(tf.reshape(stepSize, (1, 1)))
        T = tf.reshape(T, (1, -1))
        T = tf.tile(T, (B, 1))
        return model(x=x, t=T, mask=mask)
      return denoiser
    
    T = [[0.0]]
    if not(modelT is None):
      T = modelT(T)[0]
    #######################
    T = tf.reshape(T, (1, ) * len(shp) + (-1, ))
    T = tf.tile(T, tf.concat([shp, [1]], axis=0))

    def denoiser(x, mask, stepSize):
      return model(x=x, t=T, mask=mask)
    return denoiser

  def reverse(self, value, denoiser, modelT=None, sampling=None):
    if isinstance(value, tuple):
      value = self._sourceDistributionSampler(value[0])

    shp = tf.shape(value)[:-1]
    res = self._dynamicAR(
      values=value,
      model=self._makeDenoiser(denoiser, modelT, shp),
      sampling=sampling
    )
    tf.assert_equal(tf.shape(res), tf.shape(value))
    return res
  
  def calculate_loss(self, x_hat, predicted):
    B = tf.shape(predicted)[0]
    tf.assert_equal(tf.shape(predicted), (B, self.out_channels))

    return self._derivativeEncoder.calculate_loss(
      x0=x_hat['x0'], xT=x_hat['xT'], T=x_hat['T'],
      predicted=tf.reshape(predicted, (B, -1, self.channels))
    )
  
  def _applyUpdate(self, xT, KPred, stepSize, dynamic=False):
    B = tf.shape(xT)[0]
    tf.assert_equal(tf.shape(xT), (B, self.channels))

    dxdt = self._derivativeEncoder.decode(KPred, stepSize)
    tf.assert_equal(tf.shape(dxdt), tf.shape(xT))

    if dynamic:
      _, dist = normVec(dxdt)
      dist = tf.reshape(dist, (B, 1))
      # scale based on the distance to the predicted value
      aDist = tf.abs(dist)
      h = stepSize
      scale = tf.where(dist <= h, aDist / h, aDist)
      scale = tf.clip_by_value(tf.square(scale), clip_value_min=h, clip_value_max=1.0)

      tf.assert_equal(tf.shape(scale), (B, 1))
      dxdt = dxdt * (1.0 + scale)
      pass

    tf.assert_equal(tf.shape(dxdt), tf.shape(xT))
    res = xT + dxdt
    # sanitize the result to be in the range [-1000, 1000]
    res = tf.clip_by_value(res, -1000, 1000)
    return res
  
  @property
  def out_channels(self):
    return self._channels * self._derivativeEncoder.levels
  
def _distribution_from_config(config):
  if 'normal' == config:
    return normal_source_distribution
  
  if isinstance(config, dict) and ('shaped' == config['name']):
    return shaped_source_distribution(config)
  
  raise ValueError('Unknown distribution')

def autoregressive_restoration_from_config(config):
  assert 'autoregressive' == config['name']

  sampling = config['sampling']
  sampling = {
    'threshold': sampling['threshold'],
    'steps limit': sampling['steps limit'],
    'step size decay': sampling['step size decay'],
    'dynamic adjustment': sampling['dynamic adjustment'],
  }

  encoding = config['encoding']
  encoder = CDerivativeEncoder(
    levels=encoding['levels'],
    scaleByCoef=encoding['scale by coefficient'],
    scaleByTime=encoding['scale by time']
  )
  return CARProcess(
    channels=config['channels'],
    extraTrainSteps=config['extra train steps'],
    sourceDistributionSampler=_distribution_from_config(config['source distribution']),
    sampling=sampling,
    derivativeEncoder=encoder,
  )