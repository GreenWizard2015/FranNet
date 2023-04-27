import tensorflow as tf
import tensorflow.keras.layers as L
from NN.utils import extractInterpolated, sMLP

class CInterpolateExtractor(tf.keras.Model):
  def __init__(self, localMixer, **kwargs):
    super().__init__(**kwargs)
    self._localMixer = localMixer(self.name + '/LocalMixer')
    return
  
  def call(self, features, pos, training=None):
    B = tf.shape(pos)[0]
    N = tf.shape(pos)[1]
    # extract latent vectors from each 2D feature map
    latent = []
    for data in features:
      # assert 2D feature map
      tf.assert_rank(data, 4)
      vectors = extractInterpolated(data, pos)
      D = vectors.shape[-1]
      # reshape to (B * N, D) and append to latent
      latent.append(tf.reshape(vectors, (B * N, D)))
      continue
    # concatenate all latent vectors and mix them
    latent = tf.concat(latent, axis=-1)
    res = self._localMixer(latent, training=training)
    tf.assert_equal(tf.shape(res)[:1], (B * N,))
    tf.assert_rank(res, 2)
    return res
# End of CInterpolateExtractor

def _local_mixer_from_config(mixer, latentDim):
  def localMixer(name):
    return tf.keras.Sequential([
      sMLP(sizes=mixer['mlp'], activation='relu', name=name + '/mlp'),
      L.Dense(latentDim, activation=mixer['final activation'], name=name + '/final')
    ], name=name)
  return localMixer

def interpolate_extractor_from_config(config, latentDim):
  localMixer = _local_mixer_from_config(config, latentDim)
  return lambda name: CInterpolateExtractor(localMixer, name=name)