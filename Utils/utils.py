import numpy as np
import tensorflow as tf
import json, os, cv2

# define cv2_imshow function if not in colab
try:
  from google.colab.patches import cv2_imshow
except:
  def cv2_imshow(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

def setGPUMemoryLimit(limit):
  for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_virtual_device_configuration(
      gpu,
      [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=limit)]
    )
    continue
  print('GPU memory limit set to %d MB' % limit)
  return

def setupGPU():
  memory_limit = os.environ.get('TF_MEMORY_ALLOCATION_IN_MB', None)
  if memory_limit is not None:
    setGPUMemoryLimit(int(memory_limit))
    pass
  return

# function to recursively merge two configs
def merge_configs(old, new):
  if isinstance(old, dict) and isinstance(new, dict):
    keys = set(list(old.keys()) + list(new.keys()))
    for key in keys:
      value = new.get(key, old.get(key))
      if key in old:
        value = merge_configs(old[key], value)

      old[key] = value
      continue
    return old
  return new

def _load_single_config(path, folder=None):
  if folder is None: folder = []
  if not isinstance(folder, list): folder = [folder]
  curFolder = os.path.dirname(path)
  if curFolder not in folder: folder = [curFolder] + folder # new list
  
  with open(path) as f:
    config = json.load(f)

  def resolve_path(path):
    if not os.path.isabs(path):
      for folderPath in folder[::-1]:
        p = os.path.join(folderPath, path)
        if os.path.exists(p): return p
      pass
    return path

  # iterate over the config and fetch the values if 'inherit' is specified
  def iterate(config):
    for key, value in config.items():
      if isinstance(value, dict): iterate(value)
      if isinstance(value, str) and value.startswith('from:'):
        filepath = resolve_path(value[5:])
        config[key] = _load_single_config(filepath, folder)
        iterate(config[key])
      continue
    # if 'inherit' is specified, fetch the values from the inherited config
    # should be done after the iteration to avoid overriding the values
    if 'inherit' in config:
      filepath = resolve_path(config['inherit'])
      inhConfig = _load_single_config(filepath, folder)
      config.pop('inherit')
      # update the config with the inherited values
      for key, value in inhConfig.items():
        config[key] = merge_configs(value, config.get(key)) if key in config else value
        continue
      return iterate(config)
    return
  
  iterate(config)
  return config

def load_config(pathOrList, folder):
  if isinstance(pathOrList, str): return _load_single_config(pathOrList, folder)
  config = {}
  for path in pathOrList:
    config = merge_configs(config, _load_single_config(path, folder))
    continue
  return config

# helper function to create a masking function from config for the dataset
def masking_from_config(config):
  name = config['name'].lower()
  if 'grid' == name:
    size = config['size']
    total = size * size
    minC = config['min']
    maxC = config['max']
    if maxC < 0: maxC = total + maxC
    if total <= maxC: maxC = total - 1
    maskValue = config.get('mask value', 0.0)

    def _applyMasking_(src, img):
      B = tf.shape(src)[0]
      # number of masked cells per image
      maskedCellsN = tf.random.uniform((B, 1), minC, maxC + 1, dtype=tf.int32)
      # generate probability mask for each image
      mask = tf.random.uniform((B, total), 0.0, 1.0)
      # get sorted indices of the mask
      cellsOrdered = tf.argsort(mask, axis=-1, direction='DESCENDING')
      # get value of maskedCellsN-th element in each row
      indices = tf.gather(cellsOrdered, maskedCellsN, batch_dims=1)
      threshold = tf.gather(mask, indices, batch_dims=1)
      tf.assert_equal(tf.shape(threshold), (B, 1))
      # make binary mask, where 1 means NOT masked
      mask = tf.cast(mask <= threshold, tf.float32)
      # reshape mask to (B, size, size, 1)
      mask = tf.reshape(mask, (B, size, size, 1))
      # scale mask to image size
      imageSize = tf.shape(src)[1:3]
      mask = tf.image.resize(mask, imageSize, method='nearest')
      # apply mask to source image
      src = src * mask + (1.0 - mask) * maskValue
      return (src, img)
    return _applyMasking_
  
  raise ValueError('Unknown masking name: %s' % name)