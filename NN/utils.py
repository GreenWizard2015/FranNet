import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
import tensorflow.keras.layers as L
from NN.CCoordsEncodingLayer import CCoordsEncodingLayer

def shuffleBatch(batch):
  indices = tf.range(tf.shape(batch)[0])
  indices = tf.random.shuffle(indices)
  return tf.gather(batch, indices)

def sample_halton_sequence(shape, dim):
  num_results = tf.math.reduce_prod(shape[1:])
  B = shape[0]
  samples = tf.map_fn(
    lambda _: tfp.mcmc.sample_halton_sequence(dim, num_results=num_results, randomized=True),
    tf.range(B), dtype=tf.float32
  )
  tf.assert_equal(tf.shape(samples), (B, num_results, dim))
  return tf.reshape(samples, tf.concat([shape, (dim,)], axis=-1))

def normVec(x):
  V, L = tf.linalg.normalize(x, axis=-1)
  V = tf.where(tf.math.is_nan(V), 0.0, V)
  return(V, L)

def ensure4d(src):
  return tf.reshape(src, tf.concat([tf.shape(src)[:3], (-1,)], axis=-1)) # 3d/4d => 4d

def extractInterpolated(data, pos):
  '''
    0, 0 - very top left corner
    1, 1 - very bottom right corner
  '''
  s = tf.shape(data)
  hwOld = tf.cast(tf.stack([s[1], s[2]]), pos.dtype)
  # interpolate_bilinear interprets (0, 0) as the center of the top left pixel
  # interpolate_bilinear interprets (1, 1) as the center of the bottom right pixel
  # so we need to add padding to the data and shift the coordinates
  data = tf.pad(data, ((0, 0), (1, 1), (1, 1), (0, 0)), mode='CONSTANT', constant_values=0.0)
  hw = tf.cast(tf.stack([s[1] + 2, s[2] + 2]), pos.dtype) # new height and width
  # update the coordinates
  # old (0, 0) => new (1 / hw, 1 / hw)
  # old (1, 1) => new (1 - 1 / hw, 1 - 1 / hw)
  d = 1.0 / hw
  pos = pos * (hwOld / hw) # scale (0, 0) => (1, 1)
  pos = (d / 2.0) + pos # shift (0, 0) => (1 / hw, 1 / hw)
  coords = pos * hw
  return tfa.image.interpolate_bilinear(data, coords, indexing='xy')

# create a grid of coordinates in [0, 1] x [0, 1] with shape (width*width, 2)
# each point in the grid is the center of a pixel
def flatCoordsGridTF(width):
  d = 1.0 / tf.cast(width, tf.float32)
  xy = (tf.range(width, dtype=tf.float32) * d) + (d / 2.0)
  coords = tf.meshgrid(xy, xy)
  return tf.concat([tf.reshape(x, (-1, 1)) for x in coords], axis=-1)
################
class CFlatCoordsEncodingLayer(tf.keras.layers.Layer):
  def __init__(self, N=32, **kwargs):
    super().__init__(**kwargs)
    self._enc = CCoordsEncodingLayer(N)
    return

  def call(self, x):
    B = tf.shape(x)[0]
    tf.assert_equal(tf.shape(x)[:-1], (B, ))
    x = tf.cast(x, tf.float32)[..., None]
    return self._enc(x)[:, 0]
#######################################
class sMLP(tf.keras.layers.Layer):
  def __init__(self, sizes, activation='linear', dropout=0.05, **kwargs):
    super().__init__(**kwargs)
    layers = []
    for i, sz in enumerate(sizes):
      if 0.0 < dropout:
        layers.append(L.Dropout(dropout, name='%s/dropout-%i' % (self.name, i)))
      layers.append(L.Dense(sz, activation=activation, name='%s/dense-%i' % (self.name, i)))
      continue
    self._F = tf.keras.Sequential(layers, name=self.name + '/F')
    return
  
  def call(self, x, **kwargs):
    return self._F(x, **kwargs)

#######################################
# generate a sequence of steps for diffusion process
# returns (steps, prevSteps) ordered from the last to the first
def make_steps_sequence(startStep, endStep, config):
  tf.debugging.assert_greater_equal(startStep, endStep, message='startStep must be >= endStep')
  name = config['name'].lower() if isinstance(config, dict) else config.lower()
  if 'uniform' == name:
    # TODO: add support for specifying the number of steps, instead of the step size
    return make_uniform_steps_sequence(startStep, endStep, config['K'])
  
  if 'quadratic' == name:
    return make_quadratic_steps_sequence(startStep, endStep)
  
  raise NotImplementedError('Unknown steps sequence name: {}'.format(name))

def make_uniform_steps_sequence(startStep, endStep, K):
  # should always include the startStep, endStep + 1 and endStep
  steps = tf.range(endStep + 2, startStep - 1, K, dtype=tf.int32)[::-1]
  steps = tf.concat([[startStep - 1], steps, [endStep + 1]], axis=0)
  prevSteps = tf.concat([steps[1:], [endStep]], axis=0)
  return steps, prevSteps

def make_quadratic_steps_sequence(startStep, endStep):
  N = tf.cast(tf.abs(startStep - endStep), tf.float32)
  logN = tf.math.ceil(tf.math.log(N) / tf.math.log(2.0))
  logN = tf.cast(logN, tf.int32)
  steps = tf.range(logN, dtype=tf.int32)
  steps = endStep + tf.pow(2, steps)

  steps = tf.cast(steps, tf.int32)[::-1]
  steps = tf.concat([[startStep - 1], steps], axis=0)
  steps = tf.unique(steps).y # i'm too lazy to write proper code for this :D
  prevSteps = tf.concat([steps[1:], [endStep]], axis=0)
  return steps, prevSteps

def is_namedtuple(obj) -> bool:
  return (
    isinstance(obj, tuple) and
    hasattr(obj, '_asdict') and
    hasattr(obj, '_fields')
  )