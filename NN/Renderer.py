import tensorflow as tf
from NN.RestorationModel import restorationModel_from_config
  
'''
Performs batched inverse restorator process to generate V from latents and coords
'''
class Renderer(tf.keras.Model):
  def __init__(self, restorationModel, batch_size=16 * 64 * 1024, **kwargs):
    super().__init__(**kwargs)
    self._batchSize = batch_size
    self._restorator = restorationModel
    return

  def call(self, **params):
    res = self._restorator(**params)
    return res

  def train_step(self, **params):
    return self._restorator.train_step(**params)

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
      V = self._restorator.reverse(**data, training=training)
      C = tf.shape(V)[-1]
      res = res.write(i, tf.reshape(V, (B, stepBy, C)))
      continue
    #################
    index = NBatches * stepBy

    data = ittr(index, N - index)
    V = self._restorator.reverse(**data, training=training)
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
# End of Renderer class

def renderer_from_config(config):
  return Renderer(
    restorationModel=restorationModel_from_config(config['restoration model']),
    batch_size=config.get('batch_size', 16 * 64 * 1024),
  )