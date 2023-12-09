import tensorflow as tf
from .utils import extractInterpolated, ensure4d, sample_halton_sequence, generateSquareGrid
from .CBaseModel import CBaseModel

class CNerf2D(CBaseModel):
  def __init__(self, 
    encoder, renderer, restorator,
    samplesN=256, trainingSampler='uniform',
    shiftedSamples=None,
    trainingLoss=None,
    **kwargs
  ):
    super().__init__(**kwargs)
    self._encoder = encoder
    self._renderer = renderer
    self._restorator = restorator
    self.samplesN = samplesN
    self._bindShiftedSamples(shiftedSamples)
    self._bindTrainingSampler(trainingSampler)
    self._bindTrainingLoss(trainingLoss)
    return
  
  def _bindTrainingLoss(self, trainingLoss):
    self._lossParams = dict() # use default loss parameters
    if trainingLoss is None: return
    # validate training loss
    assert callable(trainingLoss), "training loss must be callable"
    self._lossParams = dict(lossFn=trainingLoss)
    return
  
  def _bindShiftedSamples(self, shiftedSamples):
    self._shiftedSamples = shiftedSamples
    if shiftedSamples is None: return
    # validate shifted samples config structure if it is present
    assert isinstance(shiftedSamples, dict), "shifted samples must be a dict"
    assert 'kind' in shiftedSamples, "shifted samples must have 'kind' key"
    assert shiftedSamples['kind'] in ['normal', 'uniform'], "shifted samples kind must be 'normal' or 'uniform'"
    assert 'fraction' in shiftedSamples, "shifted samples must have 'fraction' key"
    assert (0.0 <= shiftedSamples['fraction']) and (shiftedSamples['fraction'] <= 1.0), "shifted samples fraction must be in [0, 1]"
    return

  def _bindTrainingSampler(self, trainingSampler):
    samplers = {
      'uniform': tf.random.uniform,
      'halton': lambda shape: sample_halton_sequence(shape[:-1], shape[-1])
    }
    assert trainingSampler in samplers, f'Unknown training sampler ({trainingSampler})'
    self._trainingSampler = samplers[trainingSampler]
    return

  def _getTrainingTargets(self, srcPos, B, N):
    # TODO: clarify this, make investigation
    if self._shiftedSamples:
      mainFraction = 1.0 - self._shiftedSamples['fraction']
      NMain = tf.floor(tf.cast(N, tf.float32) * tf.cast(mainFraction, tf.float32))
      NMain = tf.cast(NMain, tf.int32)
      augmentedN = N - NMain
      # for second part of the points srcPos stays the same, but targetPos is shifted
      shifts = None
      if 'normal' == self._shiftedSamples['kind']:
        shifts = tf.random.normal((B, augmentedN, 2), stddev=self._shiftedSamples['stddev'])
      if 'uniform' == self._shiftedSamples['kind']:
        shifts = tf.random.uniform((B, augmentedN, 2)) - srcPos[:, :augmentedN]

      mainPoints = srcPos[:, :NMain]
      # use some points from the main part of the grid, so that they have the same latent vector
      shiftedPoints = srcPos[:, :augmentedN] + shifts
      srcPos = tf.concat([mainPoints, shiftedPoints], axis=1)
      srcPos = tf.clip_by_value(srcPos, 0.0, 1.0)
      pass
    tf.assert_equal(tf.shape(srcPos), (B, N, 2))
    return srcPos

  def _trainingData(self, encodedSrc, dest):
    B, C, N = tf.shape(dest)[0], tf.shape(dest)[-1], self.samplesN
    srcPos = self._trainingSampler((B, N, 2))
    # obtain latent vector for each sampled position
    latents = self._encoder.latentAt(encoded=encodedSrc, pos=srcPos, training=True)
    tf.assert_equal(tf.shape(latents)[:1], (B * N,))

    targetPos = self._getTrainingTargets(srcPos, B, N)
    x0 = extractInterpolated(dest, targetPos)
    return(
      tf.reshape(x0, (B * N, C)),
      tf.reshape(targetPos, (B * N, 2)),
      tf.reshape(latents, (B * N, -1))
    )
  
  def train_step(self, data):
    (src, dest) = data
    src = ensure4d(src)
    dest = ensure4d(dest)
    
    with tf.GradientTape() as tape:
      encodedSrc = self._encoder(src=src, training=True)
      x0, queriedPos, latents = self._trainingData(encodedSrc, dest)
      loss = self._restorator.train_step(
        x0=self._converter.convert(x0), # convert to the target format
        model=lambda T, V: self._renderer(
          latents=latents, pos=queriedPos,
          T=T, V=V,
          training=True
        ),
        **self._lossParams
      )
      
    self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
    self._loss.update_state(loss)
    return self.metrics_to_dict(self._loss)
  
  def test_step(self, images):
    (src, dest) = images
    src = ensure4d(src)
    dest = ensure4d(dest)
    # call the model itself to obtain the reconstructed image in the proper format
    reconstructed = self(src, size=tf.shape(dest)[1], training=False)
    return self._testMetrics(dest, reconstructed)

  #####################################################
  @tf.function
  def _inference(
    self, src, pos, 
    batchSize=None, reverseArgs=None, initialValues=None,
    sampleShape=None
  ):
    if reverseArgs is None: reverseArgs = {}
    encoderParams = reverseArgs.get("encoder", {})

    N = tf.shape(pos)[0]
    tf.assert_equal(tf.shape(pos), (N, 2), "pos must be a 2D tensor of shape (N, 2)")

    src = ensure4d(src)
    B = tf.shape(src)[0]
    encoded = self._encoder(src, training=False, params=encoderParams)

    if initialValues is not None:
      C = tf.shape(initialValues)[-1]
      tf.assert_equal(tf.shape(initialValues)[:1], (B, ))
      initialValues = tf.reshape(initialValues, (B, N, C))

    def getChunk(ind, sz):
      posC = pos[ind:ind+sz]
      sz = tf.shape(posC)[0]
      flatB = B * sz

      # same coordinates for all images in the batch
      posC = tf.tile(posC, [B, 1])
      tf.assert_equal(tf.shape(posC), (flatB, 2))

      latents = self._encoder.latentAt(
        encoded=encoded,
        pos=tf.reshape(posC, (B, sz, 2)),
        training=False, params=encoderParams
      )
      tf.assert_equal(tf.shape(latents)[:1], (flatB,))
      value = (flatB, )
      if initialValues is not None:
        value = initialValues[:, ind:ind+sz, :]
        tf.assert_equal(tf.shape(value), (B, sz, C))
        value = tf.reshape(value, (flatB, C))
        pass
      return dict(latents=latents, pos=posC, reverseArgs=reverseArgs, value=value)

    probes = self._renderer.batched(ittr=getChunk, B=B, N=N, batchSize=batchSize, training=False)
    if sampleShape is not None:
      C = tf.shape(probes)[-1]
      fullShape = tf.concat([[B], sampleShape, [C]], axis=0)
      probes = tf.reshape(probes, fullShape)
      pass
    # convert to the proper format
    probes = self._converter.convertBack(probes)
    return probes
  
  @tf.function
  def call(self, 
    src,
    size=32, scale=1.0, shift=0.0, # required be a default arguments for building the model
    pos=None,
    batchSize=None, # renderers batch size
    initialValues=None, # initial values for the restoration process
    reverseArgs=None,
  ):
    sampleShape = None
    if pos is None:
      pos = generateSquareGrid(size, scale, shift)
      sampleShape = (size, size)
    else:
      sampleShape = (tf.shape(pos)[0], )

    return self._inference(
      src=src, pos=pos,
      batchSize=batchSize,
      reverseArgs=reverseArgs,
      initialValues=initialValues,
      sampleShape=sampleShape
    )
  
  def get_input_shape(self):
    return self._encoder.get_input_shape()