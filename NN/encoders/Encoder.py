import tensorflow as tf
import tensorflow.keras.layers as L
from NN.utils import sMLP

def conv_block_params_from_config(config):
  defaultConvParams = {
    'kernel size': config.get('kernel size', 3),
    'activation': config.get('activation', 'relu')
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
    'activation': config.get('final activation', defaultConvParams['activation'])
  }
  return convBefore + [lastConvParams]
  
def conv_block_from_config(data, config, defaults):
  config = {**defaults, **config} # merge defaults and config
  convParams = conv_block_params_from_config(config)
  # apply convolutions to the data
  for parameters in convParams:
    data = L.Conv2D(
      filters=parameters['channels'],
      padding='same',
      kernel_size=parameters['kernel size'],
      activation=parameters['activation']
    )(data)
    continue
  return data

'''
Simple encoder that takes image as input and returns corresponding latent vector with intermediate representations
'''
def createEncoderHead(
  imgWidth, channels, downsampleSteps, latentDim, 
  ConvBeforeStage, ConvAfterStage, 
  localContext, globalContext,
  name
):
  assert isinstance(downsampleSteps, list) and (0 < len(downsampleSteps)), 'downsampleSteps must be a list of integers'
  data = L.Input(shape=(imgWidth, imgWidth, channels))
  
  res = data
  res = L.BatchNormalization()(res)
  intermediate = []
  for sz in downsampleSteps:
    res = L.Conv2D(sz, 3, strides=2, padding='same', activation='relu')(res)
    for _ in range(ConvBeforeStage):
      res = L.Conv2D(sz, 3, padding='same', activation='relu')(res)

    # local context
    if not(localContext is None):
      intermediate.append(
        conv_block_from_config(
          data=res, config=localContext, defaults={
            'channels': sz,
            'channels last': latentDim, # last layer should have latentDim channels
          }
        )
      )
    ################################
    for _ in range(ConvAfterStage):
      res = L.Conv2D(sz, 3, padding='same', activation='relu')(res)
    continue

  if not(globalContext is None): # global context
    res = conv_block_from_config(
      data=res, config=globalContext, defaults={
        'conv before': 0, # by default, no convolutions before the last layer
      }
    )
    latent = L.Flatten()(res)
    context = sMLP(sizes=globalContext['mlp'], activation='relu', name=name + '/globalMixer')(latent)
    context = L.Dense(latentDim, activation=globalContext['final activation'])(context)
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
  def __init__(self, imgWidth, channels, head, extractor, **kwargs):
    super().__init__(**kwargs)
    self._imgWidth = imgWidth
    self._channels = channels

    self._encoderHead = head(self.name + '/EncoderHead')
    self._extractor = extractor(self.name + '/Extractor')
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

    tf.assert_equal(tf.shape(context)[:-1], tf.shape(localCtx)[:-1])

    res = tf.concat([context, localCtx], axis=-1)
    tf.assert_equal(tf.shape(res)[:-1], (B * N,))
    # just to make sure that the shape is correctly inferred
    res = tf.ensure_shape(res, (None, context.shape[-1] + localCtx.shape[-1]))
    return res
  
  def get_input_shape(self):
    return (None, self._imgWidth, self._imgWidth, self._channels)
# End of CEncoder
