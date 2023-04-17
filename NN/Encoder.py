import tensorflow as tf
import tensorflow.keras.layers as L
from NN.utils import sMLP, extractInterpolated

'''
Simple encoder that takes image as input and returns corresponding latent vector with intermediate representations
'''
def createEncoderHead(
  imgWidth, channels, downsampleSteps, latentDim,
  localContext, globalContext,
  name
):
  assert(
    isinstance(downsampleSteps, list) and (0 < len(downsampleSteps)),
    'downsampleSteps must be a list of integers'
  )
  data = L.Input(shape=(imgWidth, imgWidth, channels))
  
  res = data
  res = L.BatchNormalization()(res)
  intermediate = []
  for sz in downsampleSteps:
    res = L.Conv2D(sz, 3, strides=2, padding='same', activation='relu')(res)

    # local context
    localCtx = res
    KSize = localContext['kernel size']
    for i in range(localContext['conv before']):
      localCtx = L.Conv2D(sz, KSize, padding='same', activation='relu')(localCtx)
      
    intermediate.append(
      L.Conv2D(latentDim, KSize, padding='same', activation=localContext['final activation'])(localCtx)
    )
    ################################
    res = L.Conv2D(sz, 3, padding='same', activation='relu')(res)
    res = L.Conv2D(sz, 3, padding='same', activation='relu')(res)
    continue

  # global context
  res = L.Conv2D(globalContext['channels'], globalContext['kernel size'], padding='same', activation='relu')(res)
  latent = L.Flatten()(res)
  context = sMLP(sizes=globalContext['mlp'], activation='relu', name=name + '/globalMixer')(latent)
  context = L.Dense(latentDim, activation=globalContext['final activation'])(context)

  return tf.keras.Model(
    inputs=[data], 
    outputs={
      'context': context,
      'intermediate': intermediate
    },
    name=name
  )

class CEncoder(tf.keras.Model):
  def __init__(self, 
    imgWidth, channels,
    head,
    extractMethod,
    combineMethod,
    localMixer,
    **kwargs
  ):
    assert extractMethod in ['interpolate'], 'Unknown local context extraction method'
    super().__init__(**kwargs)
    self._imgWidth = imgWidth
    self._channels = channels
    self._extractMethod = extractMethod
    self._combine = self._getCombineMethod(combineMethod)

    self._encoderHead = head(self.name + '/EncoderHead')
    self._localMixer = localMixer(self.name + '/LocalMixer')
    self._contextDropout = L.SpatialDropout1D(0.2)
    return

  def call(self, src, training=None):
    return self._encoderHead(src, training=training)
  
  def _localContext(self, encoded, pos, training=None):
    B = tf.shape(pos)[0]
    N = tf.shape(pos)[1]
    # extract latent vectors from each intermediate representation
    latent = [
      tf.reshape(extractInterpolated(data, pos), (B * N, data.shape[-1]))
      for data in encoded['intermediate']
    ]
    # concatenate all latent vectors and mix them
    latent = tf.concat(latent, axis=-1)
    localCtx = self._localMixer(latent, training=training)
    return localCtx

  def latentAt(self, encoded, pos, training=None):
    B = tf.shape(pos)[0]
    N = tf.shape(pos)[1]
    tf.assert_equal(tf.shape(pos), (B, N, 2))
    
    localCtx = self._localContext(encoded, pos, training=training)
    M = localCtx.shape[-1]
    context = encoded['context']
    tf.assert_equal(tf.shape(context), (B, M))
    tf.assert_equal(tf.shape(localCtx), (B * N, M))
    
    # apply spatial dropout to each context separately
    context = self._contextDropout(context[None], training=training)[0]
    localCtx = self._contextDropout(localCtx[None], training=training)[0]
    return self._combine(
      context=context, localCtx=localCtx,
      B=B, N=N, M=M
    )
  
  def _getCombineMethod(self, method):
    def _combine_method_add(context, localCtx, B, N, M):
      localCtx = tf.reshape(localCtx, (B, N, M))
      res = context[:, None] + localCtx
      return tf.reshape(res, (B * N, M))
    
    def _combine_method_concat(context, localCtx, B, N, M):
      context = tf.repeat(context, N, axis=0)
      res = tf.concat([context, localCtx], axis=-1)
      return tf.reshape(res, (B * N, 2 * M))
    
    if method == 'add': return _combine_method_add
    if method == 'concat': return _combine_method_concat
    raise ValueError(f'Unknown combine method: {method}')
  
  def get_input_shape(self):
    return (None, self._imgWidth, self._imgWidth, self._channels)
  
def encoder_from_config(config):
  if 'basic' == config['name']:
    imgWidth = config['image size']
    latentDim = config['latent dimension']
    headConfig = config['head']
    head = lambda name: createEncoderHead(
      imgWidth=imgWidth,
      channels=config['channels'],
      downsampleSteps=headConfig['downsample steps'],
      latentDim=latentDim,
      localContext=headConfig['local context'],
      globalContext=headConfig['global context'],
      name=name
    )

    mixer = config['contexts mixer']
    def localMixer(name):
      return tf.keras.Sequential([
        sMLP(sizes=mixer['mlp'], activation='relu', name=name + '/mlp'),
        L.Dense(latentDim, activation=mixer['final activation'], name=name + '/final')
      ], name=name)
    
    return CEncoder(
      imgWidth=imgWidth,
      channels=config['channels'],
      head=head,
      extractMethod=mixer['extract method'],
      combineMethod=mixer['combine method'],
      localMixer=localMixer,
    )
  
  raise ValueError(f"Unknown encoder name: {config['name']}")
 