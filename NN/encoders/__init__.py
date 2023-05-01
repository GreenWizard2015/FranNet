from .Encoder import CEncoder, createEncoderHead
from .extractors import extractor_from_config
import tensorflow as tf

def _getCombineMethod(method):
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
    return CEncoder(
      imgWidth=imgWidth,
      channels=config['channels'],
      head=head,
      extractor=extractor_from_config(mixer['extractor'], latentDim),
      combiner=_getCombineMethod(mixer['combine method']),
      contextDropout=config['context dropout'],
    )
  
  raise ValueError(f"Unknown encoder name: {config['name']}")
 