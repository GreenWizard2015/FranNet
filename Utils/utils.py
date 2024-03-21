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

def dumb_deepcopy(obj):
  obj = json.dumps(obj)
  obj = json.loads(obj)
  return obj

# function to recursively merge two configs
def merge_configs(old, new):
  if isinstance(old, dict) and isinstance(new, dict):
    old = dumb_deepcopy(old)
    keys = set(list(old.keys()) + list(new.keys()))
    for key in keys:
      value = new.get(key, old.get(key))
      if key in old:
        value = merge_configs(old[key], value)

      old[key] = value
      continue
    return old
  return dumb_deepcopy(new)

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

def withMoveField(config):
  def moveField(oldPath, newPath):
    # get old path value
    old = config
    for key in oldPath[:-1]:
      if key not in old: return
      old = old[key]
      continue
    if oldPath[-1] not in old: return
    value = old.pop(oldPath[-1])
    # create new path
    new = config
    for key in newPath[:-1]:
      if key not in new: new[key] = {}
      new = new[key]
      continue
    new[newPath[-1]] = value # set new value
    return
  return moveField

def upgrade_configs_structure(config):
  moveField = withMoveField(config)
  # convert old configs to new formats
  moveField(
    ['model', 'nerf', 'samplesN'],
    ['dataset', 'train', 'subsample', 'N'],
  )
  moveField(
    ['model', 'nerf', 'training sampler'],
    ['dataset', 'train', 'subsample', 'sampling'],
  )
  return config

def load_config(pathOrList, folder, upgrade=True):
  if isinstance(pathOrList, str): return _load_single_config(pathOrList, folder)
  config = {}
  for path in pathOrList:
    config = merge_configs(config, _load_single_config(path, folder))
    continue

  if upgrade: config = upgrade_configs_structure(config)
  return config

# helper function to create a masking function from config for the dataset
def _applyMasking_helper(src, minC, maxC, maskValue, size):
  total = size * size
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
  return src * mask + (1.0 - mask) * maskValue

def _grid_from_config(config):
  # convert config values to tf tensor (?, 3) int32
  def _to_params(sz):
    total = sz * sz
    minC = config['min']
    maxC = config['max']

    if isinstance(maxC, float): # percentage
      assert 0.0 <= maxC <= 1.0, 'Invalid min/max values for grid masking'
      maxC = int(maxC * total)
    else:
      if maxC < 0: maxC = total + maxC
      if total <= maxC: maxC = total - 1
      pass

    if isinstance(minC, float): # percentage
      assert 0.0 <= minC <= 1.0, 'Invalid min/max values for grid masking'
      minC = int(minC * total)
      
    assert 0 <= minC <= maxC <= total, 'Invalid min/max values for grid masking'
    return(sz, minC, maxC)
  # End of _to_params
  size = config['size']
  if not isinstance(size, list): size = [size]
  res = [_to_params(sz) for sz in size]
  return tf.constant(res, dtype=tf.int32)

def masking_from_config(config):
  name = config['name'].lower()
  if 'grid' == name:
    params = _grid_from_config(config)
    maskValue = config.get('mask value', 0.0)
    def _applyMasking(src, YData):
      idx = tf.random.uniform((), 0, tf.shape(params)[0], dtype=tf.int32)
      P = tf.gather(params, idx)
      src = _applyMasking_helper(
        src,
        maskValue=maskValue,
        minC=P[1], maxC=P[2],
        size=P[0]
      )
      return (src, YData)
    return _applyMasking
  
  raise ValueError('Unknown masking name: %s' % name)

def CFakeObject(**kwargs):
  # create a namedtuple with the given kwargs
  from collections import namedtuple
  return namedtuple('CFakeObject', kwargs.keys())(**kwargs)

# Ugly static class to load/save json
class JSONHelper:
  @staticmethod
  def load(path):
    assert os.path.exists(path), 'File not found: %s' % path
    with open(path, 'r') as f:
      return json.load(f)
    return
  
  @staticmethod
  def save(path, data):
    with open(path, 'w') as f:
      json.dump(data, f, indent=2)
    return
  