import tensorflow as tf
import tensorflow.keras.layers as L
from NN.utils import sMLP
from NN.encoding import CCoordsGridLayer, CCoordsEncodingLayer
from NN.layers import MixerConvLayer, Patches, TransformerBlock
from Utils.utils import dumb_deepcopy

def block_params_from_config(config):
  layers = config.get('layers', None)
  if not(layers is None): return layers

  defaultConvParams = {
    'kernel size': config.get('kernel size', 3),
    'activation': config.get('activation', 'relu'),
    'name': config.get('name', 'Conv2D'),
  }
  convBefore = config['conv before']
  # if convBefore is an integer, then it's the same for all layers
  if isinstance(convBefore, int):
    convParams = { 'channels': config['channels'], **defaultConvParams }
    convBefore = [convParams] * convBefore # repeat the same parameters
    pass
  assert isinstance(convBefore, list), 'convBefore must be a list'
  # if convBefore is a list of integers, then each integer is the number of channels
  if (0 < len(convBefore)) and isinstance(convBefore[0], int):
    convBefore = [ {'channels': sz, **defaultConvParams} for sz in convBefore ]
    pass

  # add separately last layer
  lastConvParams = {
    'channels': config.get('channels last', config['channels']),
    'kernel size': config.get('kernel size last', defaultConvParams['kernel size']),
    'activation': config.get('final activation', defaultConvParams['activation']),
    'name': config.get('last name', 'Conv2D'),
  }
  return convBefore + [lastConvParams]
  
def conv_block_from_config(data, config, defaults, name='CB'):
  config = {**defaults, **config} # merge defaults and config
  convParams = block_params_from_config(config)
  # apply convolutions to the data
  for i, parameters in enumerate(convParams):
    parameters = dumb_deepcopy(parameters)
    Name = parameters.get('name', 'Conv2D')
    if 'Conv2D' == Name:
      data = L.Conv2D(
        filters=parameters['channels'],
        padding='same',
        kernel_size=parameters['kernel size'],
        activation=parameters['activation'],
        name='%s/conv-%d' % (name, i)
      )(data)
      continue

    if 'MLP Mixer' == Name:
      data = MixerConvLayer(
        token_mixing=parameters.get('token mixing', 512),
        channel_mixing=parameters.get('channel mixing', 512),
        name='%s/conv-mixer-%d' % (name, i)
      )(data)
      continue

    if 'Patches' == Name:
      data = Patches(
        patch_size=parameters['patch size'],
        name='%s/patches-%d' % (name, i)
      )(data)
      continue

    if 'CoordsGrid' == Name:
      parameters = {k: v for k, v in parameters.items() if k not in ['name']}
      parameters['name'] = '%s/coordsGrid-%d' % (name, i)
      data = CCoordsGridLayer(
        CCoordsEncodingLayer(
          N=parameters.get('N', 32),
          **parameters
        ),
        name='%s/coordsGrid-%d' % (name, i)
      )(data)
      continue

    if 'Transformer' == Name:
      parameters = {k: v for k, v in parameters.items()}      
      parameters['name'] = '%s/transformer-%d' % (name, i)
      parameters['intermediate_dim'] = parameters.pop('intermediate dim', 512)
      parameters['num_heads'] = parameters.pop('num heads', 8)
      data = TransformerBlock(**parameters)(data)
      continue

    if 'Reshape' == Name:
      shape = list(parameters['shape'])
      for j, sz in enumerate(shape):
        if sz <= -2:
          sz = data.shape[sz + 1]
        shape[j] = sz
        continue
      data = L.Reshape(
        shape,
        name='%s/reshape-%d' % (name, i)
      )(data)
      continue

    if 'MLP' == Name:
      parameters['name'] = '%s/mlp-%d' % (name, i)
      data = sMLP(**parameters)(data)
      continue
    
    raise NotImplementedError('Unknown layer: {}'.format(Name))
  return data

def _createGCMv2(dataShape, config, latentDim, name):
  data = L.Input(shape=dataShape)

  res = data
  for i, blockConfig in enumerate(config['downsample steps']):
    # downsample
    res = L.Conv2D(
      filters=blockConfig['channels'],
      kernel_size=blockConfig['kernel size'],
      strides=2,
      padding='same',
      activation='relu',
      name=name + '/downsample-%d' % (i + 1,)
    )(res)
    # convolutions
    for layerId in range(blockConfig['layers']):
      res = L.Conv2D(
        filters=blockConfig['channels'],
        kernel_size=blockConfig['kernel size'],
        padding='same',
        activation='relu',
        name=name + '/downsample-%d/layer-%d' % (i + 1, layerId + 1)
      )(res)
      continue
    continue

  return tf.keras.Model(inputs=[data], outputs=res, name=name)

def _createGlobalContextModel(X, config, latentDim, name):
  model = config.get('name', 'v1')
  if 'v1' == model: # simple convolutional model
    res = conv_block_from_config(
      data=X, config=config, defaults={
        'conv before': 0, # by default, no convolutions before the last layer
      }
    )
    # calculate global context
    latent = L.Flatten()(res)
    context = sMLP(sizes=config['mlp'], activation='relu', name=name + '/globalMixer')(latent)
    context = L.Dense(latentDim, activation=config['final activation'])(context)
    return context # end of 'v1' model
  
  if 'v2' == model:
    res = data = L.Input(shape=X.shape[1:])
    res = _createGCMv2(res.shape[1:], config, latentDim, name)(res)
    # calculate global context
    latent = L.Flatten()(res)
    context = sMLP(sizes=config['mlp'], activation='relu', name=name + '/globalMixer')(latent)
    context = L.Dense(latentDim, activation=config['final activation'])(context)
    model = tf.keras.Model(inputs=[data], outputs=context, name=name)
    return model(X) # end of 'v2' model
  
  raise NotImplementedError('Unknown global context model: {}'.format(model))

def _withPositionConfig(config, name):
  if config is None: return lambda x, _: x
  
  if isinstance(config, bool): config = { 'N': 32 }
  assert isinstance(config, dict), 'config must be a dictionary'

  def withPosition(x, i):
    if not config.get('stage-%d' % i, True): return x

    encoding = config.get('encoding', {})
    encoding = dict(**encoding)
    encoding['N'] = config.get('stage-%d N' % i, config.get('N', 32))
    return CCoordsGridLayer(
      CCoordsEncodingLayer(**encoding, name='%s/coordsGrid-%d/encoding' % (name, i)),
      name='%s/coordsGrid-%d' % (name, i)
    )(x)
  return withPosition

'''
Simple encoder that takes image as input and returns corresponding latent vector with intermediate representations
'''
def createEncoderHead(
  imgWidth,
  config,
  channels, downsampleSteps, latentDim, 
  ConvBeforeStage, ConvAfterStage, 
  localContext, globalContext,
  positionsConfigs,
  name
):
  assert config is not None, 'config must be a dictionary'
  assert isinstance(downsampleSteps, list) and (0 < len(downsampleSteps)), 'downsampleSteps must be a list of integers'
  data = L.Input(shape=(imgWidth, imgWidth, channels))
  
  withPosition = _withPositionConfig(positionsConfigs, name)
  res = data
  res = L.BatchNormalization()(res)
  intermediate = []
  for i, sz in enumerate(downsampleSteps):
    if config.get('use downsampling', True):
      res = L.Conv2D(sz, 3, strides=2, padding='same', activation='relu')(res)
    res = withPosition(res, i) # add position encoding if needed
    for _ in range(ConvBeforeStage):
      res = L.Conv2D(sz, 3, padding='same', activation='relu')(res)

    # local context
    if not(localContext is None):
      intermediate.append(
        conv_block_from_config(
          data=res, config=localContext, defaults={
            'channels': sz,
            'channels last': latentDim, # last layer should have latentDim channels
          },
          name='%s/intermediate-%d' % (name, i)
        )
      )
    ################################
    for _ in range(ConvAfterStage):
      res = L.Conv2D(sz, 3, padding='same', activation='relu')(res)
    continue

  if not(globalContext is None): # global context
    res = withPosition(res, len(downsampleSteps))
    # context = Patches(patch_size=16)(context)
    context = conv_block_from_config(
        data=res, config=localContext, defaults={
        'channels': sz,
        'channels last': latentDim, # last layer should have latentDim channels
        },
        name='%s/intermediate-final' % (name, )
    )
    # extract center of the context
    context = L.Lambda(lambda x: x[:, x.shape[1] // 2, x.shape[2] // 2])(context)
  else: # no global context
    # return dummy context to keep the interface consistent
    context = L.Lambda(
      lambda x: tf.zeros((tf.shape(x)[0], 1), dtype=res.dtype)
    )(res)

  return tf.keras.Model(
    inputs=[data],
    outputs={
      'intermediate': intermediate, # intermediate representations
      'context': context, # global context
    },
    name=name
  )

class CEncoder(tf.keras.Model):
  def __init__(self, imgWidth, channels, head, extractor, dropoutRate=0.25, **kwargs):
    super().__init__(**kwargs)
    self._imgWidth = imgWidth
    self._channels = channels

    self._encoderHead = head(self.name + '/EncoderHead')
    self._extractor = extractor(self.name + '/Extractor')
    # spatial 2d dropout for the local context
    self._dropoutRate = dropoutRate
    return

  def call(self, src, training=None, params=None):
    res = self._encoderHead(src, training=training)
    # ablation study of intermediate representations
    if not(params is None):
      def applyIntermediateMask(i, x):
        if params.get('no intermediate {}'.format(i + 1), False): return tf.zeros_like(x)
        return x
      
      res['intermediate'] = [applyIntermediateMask(i, x) for i, x in enumerate(res['intermediate'])]
      pass

    return res

  def latentAt(self,
    encoded, pos, training=None,
    params=None # parameters for ablation study
  ):
    if training is None:
      training = tf.keras.backend.learning_phase()
    B = tf.shape(pos)[0]
    N = tf.shape(pos)[1]
    tf.assert_equal(tf.shape(pos), (B, N, 2))
    
    # global context is always present, even if it's a dummy one
    context = encoded['context']
    context = tf.repeat(context, N, axis=0)
    tf.assert_equal(tf.shape(context)[:-1], (B * N,))
    if self._extractor is None: return context # local context is disabled
    
    # local context could be absent
    localCtx = self._extractor(encoded['intermediate'], pos, training=training)
    tf.assert_equal(tf.shape(localCtx)[:-1], (B * N,))

    # ablation study
    if not(params is None):
      noLocalCtx = params.get('no local context', False)
      noGlobalCtx = params.get('no global context', False)
      assert not(noLocalCtx and noGlobalCtx), 'can\'t drop both local and global context at the same time'

      if noLocalCtx: localCtx = tf.zeros_like(localCtx)
      if noGlobalCtx: context = tf.zeros_like(context)
      pass

    tf.assert_equal(tf.shape(context)[:-1], (B * N,))
    tf.assert_equal(tf.shape(localCtx)[:-1], (B * N,))
    if training and (0.0 < self._dropoutRate):
      msk = tf.random.uniform([B * N, 1], dtype=context.dtype)
      localCtx = tf.where(msk < self._dropoutRate, tf.zeros_like(localCtx), localCtx)
      pass
    res = tf.concat([context, localCtx], axis=-1)
    tf.assert_equal(tf.shape(res)[:-1], (B * N,))
    # just to make sure that the shape is correctly inferred
    res = tf.ensure_shape(res, (None, context.shape[-1] + localCtx.shape[-1]))
    return res
  
  def get_input_shape(self):
    return (None, self._imgWidth, self._imgWidth, self._channels)
# End of CEncoder
