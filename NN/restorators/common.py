import tensorflow as tf

def clipping_from_config(config):
  if config is None: return lambda x: x
  
  assert 'min' in config, 'clipping config must contain "min" key'
  assert 'max' in config, 'clipping config must contain "max" key'
  
  return lambda x: tf.clip_by_value(x, clip_value_min=config['min'], clip_value_max=config['max'])