import tensorflow as tf
import tensorflow.keras.layers as L
from NN.utils import sMLP

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
  def __init__(self, imgWidth, channels, head, extractor, combiner, **kwargs):
    super().__init__(**kwargs)
    self._imgWidth = imgWidth
    self._channels = channels

    self._encoderHead = head(self.name + '/EncoderHead')
    self._extractor = extractor(self.name + '/Extractor')
    self._combine = combiner
    self._contextDropout = L.SpatialDropout1D(0.2)
    return

  def call(self, src, training=None):
    return self._encoderHead(src, training=training)

  def latentAt(self, encoded, pos, training=None):
    B = tf.shape(pos)[0]
    N = tf.shape(pos)[1]
    tf.assert_equal(tf.shape(pos), (B, N, 2))
    
    localCtx = self._extractor(encoded['intermediate'], pos, training=training)
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
  
  def get_input_shape(self):
    return (None, self._imgWidth, self._imgWidth, self._channels)
# End of CEncoder
