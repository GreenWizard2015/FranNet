import tensorflow as tf
from .utils import extractInterpolated, ensure4d, flatCoordsGridTF
from .CBaseModel import CBaseModel

class CNerf2D(CBaseModel):
  def __init__(self, encoder, renderer, restorator, samplesN=256, **kwargs):
    super().__init__(**kwargs)
    self._encoder = encoder
    self._renderer = renderer
    self._restorator = restorator
    self.samplesN = samplesN
    return
  
  def train_step(self, data):
    (src, dest) = data
    src = ensure4d(src)
    dest = ensure4d(dest)
    
    B = tf.shape(src)[0]
    C = tf.shape(dest)[-1]
    N = self.samplesN # number of points sampled from each image
    with tf.GradientTape() as tape:
      pos = tf.random.uniform((B, N, 2), dtype=tf.float32)
      x0 = extractInterpolated(dest, pos)
      # obtain latent vector for each sampled position
      latents = self._encoder.latentAt(
        encoded=self._encoder(src=src, training=True),
        pos=pos,
        training=True
      )
      tf.assert_equal(tf.shape(latents)[:1], (B * N,))
      tf.assert_equal(tf.shape(pos), (B, N, 2))
      tf.assert_equal(tf.shape(x0), (B, N, C))
      # flatten the batch dimension
      pos = tf.reshape(pos, (B * N, 2))
      latents = tf.reshape(latents, (B * N, -1))

      loss = self._restorator.train_step(
        x0=tf.reshape(x0, (B * N, C)),
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
  @tf.function
  def call(self, 
    src,
    size=32, scale=1.0, shift=0.0, batchSize=None,
    reverseArgs=None
  ):
    src = ensure4d(src)
    B = tf.shape(src)[0]
    encoded = self._encoder(src, training=False)

    pos = (flatCoordsGridTF(size) * scale) + shift
    N = tf.shape(pos)[0]
    tf.assert_equal(N, size * size) # just in case
    tf.assert_equal(tf.shape(pos), (N, 2))

    def getChunk(ind, sz):
      posC = pos[ind:ind+sz]
      sz = tf.shape(posC)[0]

      # same coordinates for all images in the batch
      posC = tf.tile(posC, [B, 1])
      tf.assert_equal(tf.shape(posC), (B * sz, 2))

      latents = self._encoder.latentAt(
        encoded=encoded,
        pos=tf.reshape(posC, (B, sz, 2)),
        training=False
      )
      tf.assert_equal(tf.shape(latents)[:1], (B * sz,))
      return(latents, posC, reverseArgs)

    probes = self._renderer.batched(
      ittr=getChunk,
      B=B, N=N,
      batchSize=batchSize,
      training=False
    )
    C = tf.shape(probes)[-1]
    return tf.reshape(probes, (B, size, size, C))
  
  def get_input_shape(self):
    return self._encoder.get_input_shape()