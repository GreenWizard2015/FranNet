from .encoders import encoder_from_config
from .Decoder import decoder_from_config
from .Renderer import renderer_from_config
from .restorators import restorator_from_config
from .Nerf2D import CNerf2D
import tensorflow as tf

def _optimizer_from_config(config):
  if isinstance(config, str): return config # return optimizer name
  if isinstance(config, dict):
    if 'adam' == config['name']:
      return tf.keras.optimizers.Adam(
        learning_rate=config['learning_rate'],
      )
    
    if 'AdamW' == config['name']:
      optimizer = tf.keras.optimizers.AdamW(
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
      )
      optimizer.exclude_from_weight_decay(
        var_names=config.get('exclude_from_weight_decay', [])
      )
      return optimizer
    pass
  raise ValueError(f"Unknown optimizer config: {config}")

def _makeTrainingLoss(config):
  if config is None: return None
  if isinstance(config, str): # name of the loss
    return tf.keras.losses.get(config)
  
  if isinstance(config, dict):
    name = config['name'].lower()
    if 'huber' == name:
      return tf.keras.losses.Huber(delta=float(config['delta']))
    
    if 'pseudo huber' == name:
      delta = float(config['delta'])
      def _pseudo_huber_loss(y_true, y_pred):
        err = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
        return tf.sqrt(err + delta**2) - delta
      return _pseudo_huber_loss

  raise ValueError(f"Unknown training loss config: {config}")

def _nerf_from_config(config):
  if 'basic' == config['name']:
    nerfParams = dict(
      samplesN=config['samplesN'],
      trainingSampler=config.get('training sampler', 'uniform'),
      shiftedSamples=config.get('shifted samples', None),
      trainingLoss=_makeTrainingLoss(config.get('training loss', None)),
    )
    # If format is not specified, use BGR, because old models were trained to predict BGR
    nerfParams['format'] = config.get('format', 'bgr')
    
    return lambda encoder, renderer, restorator: CNerf2D(
      encoder=encoder,
      renderer=renderer,
      restorator=restorator,
      **nerfParams
    )
  
  raise ValueError(f"Unknown nerf name: {config['name']}")

def model_from_config(config, compile=True):
  encoder = encoder_from_config(config['encoder'])
  decoder = decoder_from_config(config['decoder'])
  restorator = restorator_from_config(config['restorator'])
  renderer = renderer_from_config(config['renderer'], decoder, restorator)
  
  nerf = _nerf_from_config(config['nerf'])(
    encoder, renderer, restorator
  )
  nerf.build(nerf.get_input_shape())

  if compile:
    nerf.compile(optimizer=_optimizer_from_config(config['optimizer']))
  return nerf

def model_to_architecture(model):
  def traverse(model, data):
    if isinstance(model, tf.keras.Model):
      data[model.name] = dataCur = {}

      try:
        params = model.count_params()
      except:
        params = 'unknown'
        pass
      dataCur['params'] = params
      dataCur['class'] = model.__class__.__name__

      for layer in model.layers:
        traverse(layer, dataCur)
        continue
    return
  
  res = {}
  traverse(model, res)
  return res