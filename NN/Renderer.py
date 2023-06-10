import tensorflow as tf
from NN.utils import CFlatCoordsEncodingLayer

'''
Performs batched inverse restorator process to generate V from latents and coords
'''
class Renderer(tf.keras.Model):
  def __init__(self, 
    decoder, restorator,
    posEncoder, timeEncoder,
    batch_size=16 * 64 * 1024,
    **kwargs
  ):
    assert posEncoder is not None, "posEncoder is not provided"
    assert timeEncoder is not None, "timeEncoder is not provided"
    super().__init__(**kwargs)
    self._batchSize = batch_size
    self._decoder = decoder
    self._restorator = restorator
    self._posEncoder = posEncoder
    self._timeEncoder = timeEncoder
    return

  # only for training and building, during inference use 'batched' method
  def call(self, latents, pos, T, V):
    pos = self._posEncoder(pos)
    T = self._timeEncoder(T)
    res = self._decoder(latents, pos, T, V)
    return res

  def _invD(self, latents, pos, reverseArgs, training):
    reverseArgs = {} if reverseArgs is None else reverseArgs
    decoderArgs = reverseArgs.get('decoder', {})
    # for ablation study of the decoder, randomize positions BEFORE encoding
    if decoderArgs.get('randomize positions', False):
      pos = tf.random.uniform(tf.shape(pos), minval=0.0, maxval=1.0)

    EPos = self._posEncoder(pos, training=training)

    def denoiser(x, t, mask=None):
      args = (latents, EPos, t, x)
      if mask is not None:
        args = (
          tf.boolean_mask(latents, mask),
          tf.boolean_mask(EPos, mask),
          tf.boolean_mask(t, mask),
          tf.boolean_mask(x, mask)
        )

      return self._decoder(*args, training=training)
    
    return self._restorator.reverse(
      value=(tf.shape(pos)[0], ),
      denoiser=denoiser,
      modelT=lambda t: self._timeEncoder(t, training=training),
      **reverseArgs
    )

  def batched(self, ittr, B, N, batchSize=None, training=False):
    batchSize = self._batchSize if batchSize is None else batchSize
    # B * stepBy <= batchSize... stepBy <= batchSize / B
    stepBy = batchSize // B
    batchSize = stepBy * B
    NBatches = N // stepBy
    res = tf.TensorArray(tf.float32, 1 + NBatches, dynamic_size=False, clear_after_read=True)
    for i in tf.range(NBatches):
      index = i * stepBy
      data = ittr(index, stepBy)
      V = self._invD(*data, training=training)
      C = tf.shape(V)[-1]
      res = res.write(i, tf.reshape(V, (B, stepBy, C)))
      continue
    #################
    index = NBatches * stepBy

    data = ittr(index, N - index)
    V = self._invD(*data, training=training)
    C = tf.shape(V)[-1]

    w = N - index
    V = tf.reshape(V, (B, w, C))
    V = tf.pad(V, [[0, 0], [0, stepBy - w], [0, 0]])
    tf.assert_equal(tf.shape(V), (B, stepBy, C))
    res = res.write(NBatches, V)
    #################
    res = res.stack() # (?, B, stepBy, C)
    tf.assert_equal(tf.shape(res)[1:], (B, stepBy, C))

    res = tf.transpose(res, (1, 0, 2, 3))
    res = tf.reshape(res, (-1, (NBatches + 1) * stepBy, C))[:, :N]
    tf.assert_equal(tf.shape(res), (B, N, C))
    return res
  
def renderer_from_config(config, decoder, restorator):
  posEncoder = None
  if 'learned' == config['position encoding']:
    posEncoder = CFlatCoordsEncodingLayer()
  ####################
  timeEncoder = None
  if 'learned' == config['time encoding']:
    timeEncoder = CFlatCoordsEncodingLayer()
  ####################
  return Renderer(
    decoder=decoder,
    restorator=restorator,
    batch_size=config.get('batch_size', 16 * 64 * 1024),
    posEncoder=posEncoder,
    timeEncoder=timeEncoder
  )