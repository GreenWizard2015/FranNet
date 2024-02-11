import tensorflow as tf
from .utils import extractInterpolated, ensure4d, generateSquareGrid
from .CBaseModel import CBaseModel

class CNerf2D(CBaseModel):
  def __init__(self, 
    encoder, renderer, restorator,
    trainingLoss=None,
    residual=False,
    extraLatents=None,
    **kwargs
  ):
    super().__init__(**kwargs)
    self._encoder = encoder
    self._renderer = renderer
    self._restorator = restorator
    self._bindTrainingLoss(trainingLoss)
    self._bindExtraLatents(extraLatents)
    self._residual = residual
    return
  
  def _bindTrainingLoss(self, trainingLoss):
    self._lossParams = dict() # use default loss parameters
    if trainingLoss is None: return
    # validate training loss
    assert callable(trainingLoss), "training loss must be callable"
    self._lossParams = dict(lossFn=trainingLoss)
    return
  
  def _bindExtraLatents(self, extraLatents):
    self._extraLatents = extraLatents
    if extraLatents is None: return
    # validate extra latents config structure if it is present
    assert isinstance(extraLatents, list), "extra latents must be a list"
    for latent in extraLatents:
      assert isinstance(latent, dict), "extra latent must be a dict"
      assert 'name' in latent, "extra latent must have 'name' key"
      continue
    return

  def _extractLatents(self, encodedSrc, positions, training=True):
    B, N = tf.shape(positions)[0], tf.shape(positions)[1]
    tf.assert_equal(tf.shape(positions), (B, N, 2))
    # obtain latent vector for each sampled position
    latents = self._encoder.latentAt(encoded=encodedSrc, pos=positions, training=training)
    tf.assert_equal(tf.shape(latents)[:1], (B * N,))
    return latents

  def _extractGrayscaled(self, img, points):
    B = tf.shape(img)[0]
    points = tf.reshape(points, (B, -1, 2))
    N = tf.shape(points)[1]
    RV = extractInterpolated(img, points)
    tf.assert_equal(tf.shape(RV)[-1], 1)
    RV = tf.repeat(RV, 3, axis=-1) # grayscale to RGB
    RV = tf.reshape(RV, (B, N, 3)) # ensure proper shape, especially for last dimension
    return RV
  
  def _withResidual(self, img, points, values, add=True):
    if not self._residual: return values

    RV = self._extractGrayscaled(img, points)
    RV = tf.reshape(RV, tf.shape(values))
    if add: return values + RV
    return values - RV # for training

  def _extractExtraLatents(self, config, src, points, latents):
    name = config['name'].lower()
    if 'grayscale' == name:
      return self._converter.convert(
        self._extractGrayscaled(src, points)
      )
    
    raise NotImplementedError(f"Unknown extra latent ({name})")
  
  def _withExtraLatents(self, latents, src, points):
    if self._extraLatents is None: return latents
    
    extraData = [
      self._extractExtraLatents(latentConfig, src, points, latents)
      for latentConfig in self._extraLatents
    ]

    C = sum([x.shape[-1] for x in extraData])
    extraData = tf.reshape( # ensure proper shape, especially for last dimension
      tf.concat(extraData, axis=-1),
      tf.concat([tf.shape(latents)[:-1], [C]], axis=0)
    )
    return tf.concat([latents, extraData], axis=-1)
  
  def train_step(self, data):
    (src, YData) = data
    src = ensure4d(src)
    x0 = YData['sampled']
    positions = YData['positions']
    
    with tf.GradientTape() as tape:
      encodedSrc = self._encoder(src=src, training=True)
      latents = self._extractLatents(encodedSrc=encodedSrc, positions=positions, training=True)
      # train the restorator
      x0 = self._withResidual(src, points=positions, values=x0, add=False)
      latents = self._withExtraLatents(latents, src=src, points=positions)
      # flatten latents and positions
      BN = tf.shape(positions)[0] * tf.shape(positions)[1]
      latents = tf.reshape(latents, (BN, tf.shape(latents)[-1]))
      positions = tf.reshape(positions, (BN, 2))
      x0 = tf.reshape(x0, (BN, tf.shape(x0)[-1]))
      # actual training step
      loss = self._restorator.train_step(
        x0=self._converter.convert(x0), # convert to the target format
        model=lambda T, V: self._renderer(
          latents=latents, pos=positions,
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

  def _createAlgorithmInterceptor(self, interceptor, image, pos):
    from NN.restorators.samplers.CWatcherWithExtras import CWatcherWithExtras
    def interceptorFactory(algorithm):
      res = interceptor(algorithm)
      residuals = None
      if self._residual:
        residuals = self._extractGrayscaled(image, pos)
        residuals = tf.reshape(residuals, (-1, tf.shape(residuals)[-1]))
        pass

      res = CWatcherWithExtras(
        watcher=res,
        converter=self._converter,
        residuals=residuals
      )
      return res
    return interceptorFactory
  #####################################################
  @tf.function
  def _inference(
    self, src, pos, 
    batchSize, reverseArgs, initialValues, encoderParams,
  ):
    B = tf.shape(src)[0]
    N = tf.shape(pos)[0]
    tf.assert_equal(tf.shape(pos), (N, 2), "pos must be a 2D tensor of shape (N, 2)")

    if initialValues is not None:
      C = tf.shape(initialValues)[-1]
      tf.assert_equal(tf.shape(initialValues)[:1], (B, ))
      initialValues = tf.reshape(initialValues, (B, N, C))

    encoded = self._encoder(src, training=False, params=encoderParams)
    def getChunk(ind, sz):
      posC = pos[ind:ind+sz]
      sz = tf.shape(posC)[0]
      flatB = B * sz

      # same coordinates for all images in the batch
      posC = tf.tile(posC, [B, 1])
      tf.assert_equal(tf.shape(posC), (flatB, 2))
      posCB = tf.reshape(posC, (B, sz, 2))

      latents = self._encoder.latentAt(
        encoded=encoded, pos=posCB, params=encoderParams,
        training=False
      )
      tf.assert_equal(tf.shape(latents)[:1], (flatB,))
      value = (flatB, )
      if initialValues is not None:
        value = initialValues[:, ind:ind+sz, :]
        value = self._converter.convert(value) # convert initial values to the proper format
        tf.assert_equal(tf.shape(value), (B, sz, C))
        value = tf.reshape(value, (flatB, C))
        pass

      # add extra latents if needed
      latents = self._withExtraLatents(latents=latents, src=src, points=posC)
      return dict(latents=latents, pos=posC, reverseArgs=reverseArgs, value=value)

    probes = self._renderer.batched(ittr=getChunk, B=B, N=N, batchSize=batchSize, training=False)
    # convert to the proper format
    probes = self._converter.convertBack(probes)
    probes = self._withResidual(
      src, values=probes,
      points=tf.tile(pos[None], [B, 1, 1])
    )
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
    src = ensure4d(src)
    B = tf.shape(src)[0]
    # precompute the output shape
    sampleShape = None
    if pos is None:
      pos = generateSquareGrid(size, scale, shift)
      sampleShape = (size, size)
    else:
      sampleShape = (tf.shape(pos)[0], )
      pass
    # prepare the reverseArgs and encoderParams
    if reverseArgs is None: reverseArgs = {}
    assert isinstance(reverseArgs, dict), "reverseArgs must be a dict"
    # extract encoder parameters from reverseArgs
    encoderParams = reverseArgs.get("encoder", {})
    reverseArgs = {k: v for k, v in reverseArgs.items() if k != 'encoder'}
    # add interceptors if needed
    if 'algorithmInterceptor' in reverseArgs:
      newParams = {k: v for k, v in encoderParams.items()}
      newParams['algorithmInterceptor'] = self._createAlgorithmInterceptor(
        interceptor=reverseArgs['algorithmInterceptor'],
        image=src, pos=tf.tile(pos[None], [B, 1, 1])
      )
      encoderParams = newParams
      # remove the interceptor from the reverseArgs
      reverseArgs = {k: v for k, v in reverseArgs.items() if k != 'algorithmInterceptor'}
      pass
    
    probes = self._inference(
      src=src, pos=pos,
      batchSize=batchSize,
      reverseArgs=reverseArgs,
      encoderParams=encoderParams,
      initialValues=initialValues
    )
    
    C = tf.shape(probes)[-1]
    fullShape = tf.concat([[B], sampleShape, [C]], axis=0)
    probes = tf.reshape(probes, fullShape)
    return probes
  
  def get_input_shape(self):
    return self._encoder.get_input_shape()