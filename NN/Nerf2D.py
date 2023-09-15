import tensorflow as tf
from .utils import extractInterpolated, ensure4d, flatCoordsGridTF
from .CBaseModel import CBaseModel
from .utils import sample_halton_sequence

class CNerf2D(CBaseModel):
  def __init__(self, 
    encoder, renderer, restorator,
    samplesN=256, trainingSampler='uniform',
    shiftedSamples=None,
    **kwargs
  ):
    super().__init__(**kwargs)
    self._encoder = encoder
    self._renderer = renderer
    self._restorator = restorator
    self.samplesN = samplesN
    self._shiftedSamples = shiftedSamples
    # validate shifted samples config structure if it is present
    if shiftedSamples is not None:
      assert isinstance(shiftedSamples, dict), "shifted samples must be a dict"
      assert 'kind' in shiftedSamples, "shifted samples must have 'kind' key"
      assert shiftedSamples['kind'] in ['normal', 'uniform'], "shifted samples kind must be 'normal' or 'uniform'"
      assert 'fraction' in shiftedSamples, "shifted samples must have 'fraction' key"
      assert (0.0 <= shiftedSamples['fraction']) and (shiftedSamples['fraction'] <= 1.0), "shifted samples fraction must be in [0, 1]"
      pass

    samplers = {
      'uniform': tf.random.uniform,
      'halton': lambda shape: sample_halton_sequence(shape[:-1], shape[-1])
    }
    assert trainingSampler in samplers, f'Unknown training sampler ({sample_halton_sequence})'
    self._trainingSampler = samplers[trainingSampler]
    return

  def _getTrainingTargets(self, srcPos, B, N):
    # TODO: clarify this, make investigation
    if self._shiftedSamples:
      mainFraction = 1.0 - self._shiftedSamples['fraction']
      NMain = tf.floor(tf.cast(N, tf.float32) * tf.cast(mainFraction, tf.float32))
      NMain = tf.cast(NMain, tf.int32)
      # for second part of the points srcPos stays the same, but targetPos is shifted
      shifts = None
      if 'normal' == self._shiftedSamples['kind']:
        shifts = tf.random.normal((B, N, 2), stddev=self._shiftedSamples['stddev'])
      if 'uniform' == self._shiftedSamples['kind']:
        shifts = tf.random.uniform((B, N, 2)) - srcPos

      return tf.concat([
        srcPos[:, :NMain], # predict sampled points
        srcPos[:, NMain:] + shifts[:, NMain:] # predict nearby points
      ], 1)
    # predict sampled points
    return srcPos

  def _trainingData(self, encodedSrc, dest):
    B = tf.shape(dest)[0]
    C = tf.shape(dest)[-1]
    N = self.samplesN # number of points sampled from each image
    
    # by default, source and target positions are the same
    srcPos = self._trainingSampler((B, N, 2))
    targetPos = self._getTrainingTargets(srcPos, B, N)

    tf.assert_equal(tf.shape(targetPos), (B, N, 2))
    tf.assert_equal(tf.shape(srcPos), (B, N, 2))

    x0 = extractInterpolated(dest, targetPos)
    # obtain latent vector for each sampled position
    latents = self._encoder.latentAt(encoded=encodedSrc, pos=srcPos, training=True)

    tf.assert_equal(tf.shape(latents)[:1], (B * N,))
    tf.assert_equal(tf.shape(targetPos), (B, N, 2))
    tf.assert_equal(tf.shape(x0), (B, N, C))
    x0 = tf.reshape(x0, (B * N, C))
    targetPos = tf.reshape(targetPos, (B * N, 2))
    latents = tf.reshape(latents, (B * N, -1))
    return(x0, targetPos, latents)
  
  def train_step(self, data):
    (src, dest) = data
    src = ensure4d(src)
    dest = ensure4d(dest)
    
    with tf.GradientTape() as tape:
      encodedSrc = self._encoder(src=src, training=True)
      x0, pos, latents = self._trainingData(encodedSrc, dest)
      loss = self._restorator.train_step(
        x0=x0,
        model=lambda T, V: self._renderer(
          latents=latents, pos=pos,
          T=T, V=V,
          training=True
        ),
      )
      
    self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
    self._loss.update_state(loss)
    return self.metrics_to_dict(self._loss)
  
  def test_step(self, images):
    (src, dest) = images
    src = ensure4d(src)
    dest = ensure4d(dest)
    
    reconstructed = self(src, size=tf.shape(dest)[1], training=False)
    return self._testMetrics(dest, reconstructed)

  #####################################################
  def _prepareGrid(self, size, scale, shift):
    # prepare coordinates
    scale = tf.cast(scale, tf.float32)
    shift = tf.cast(shift, tf.float32)
    # make sure that scale and shift 2 rank tensors 2 elements
    # (flatten them, concatenate with themselves, take first 2 elements, reshape)
    scale = tf.reshape(scale, (-1,))
    scale = tf.concat([scale, scale], axis=0)[:2]
    scale = tf.reshape(scale, (1, 2))

    shift = tf.reshape(shift, (-1,))
    shift = tf.concat([shift, shift], axis=0)[:2]
    shift = tf.reshape(shift, (1, 2))
    # prepare coordinates
    pos = flatCoordsGridTF(size)
    return (pos * scale) + shift

  @tf.function
  def inference(self, src, pos, batchSize=None, reverseArgs=None):
    if reverseArgs is None: reverseArgs = {}
    encoderParams = reverseArgs.get("encoder", {})

    N = tf.shape(pos)[0]
    tf.assert_equal(tf.shape(pos), (N, 2), "pos must be a 2D tensor of shape (N, 2)")

    src = ensure4d(src)
    B = tf.shape(src)[0]
    encoded = self._encoder(src, training=False, params=encoderParams)

    def getChunk(ind, sz):
      posC = pos[ind:ind+sz]
      sz = tf.shape(posC)[0]

      # same coordinates for all images in the batch
      posC = tf.tile(posC, [B, 1])
      tf.assert_equal(tf.shape(posC), (B * sz, 2))

      latents = self._encoder.latentAt(
        encoded=encoded,
        pos=tf.reshape(posC, (B, sz, 2)),
        training=False, params=encoderParams
      )
      tf.assert_equal(tf.shape(latents)[:1], (B * sz,))
      return(latents, posC, reverseArgs)

    return self._renderer.batched(
      ittr=getChunk,
      B=B, N=N,
      batchSize=batchSize,
      training=False
    )
  
  @tf.function
  def call(self, 
    src,
    size=32, scale=1.0, shift=0.0, # required be a default arguments for building the model
    pos=None,
    batchSize=None, # renderers batch size
    reverseArgs=None,
  ):
    B = tf.shape(src)[0]
    if pos is None:
      pos = self._prepareGrid(size, scale, shift)
      tf.assert_equal(tf.shape(pos), (size * size, 2)) # just in case
      
      probes = self.inference(
        src=src, pos=pos,
        batchSize=batchSize,
        reverseArgs=reverseArgs
      )
      C = tf.shape(probes)[-1]
      return tf.reshape(probes, (B, size, size, C))
    # if pos is not None, then we are rendering a custom grid with arbitrary shape
    probes = self.inference(
      src=src, pos=pos,
      batchSize=batchSize,
      reverseArgs=reverseArgs
    )
    tf.assert_equal(tf.shape(probes), (B, tf.shape(pos)[0], tf.shape(probes)[-1]))
    return probes
  
  def get_input_shape(self):
    return self._encoder.get_input_shape()