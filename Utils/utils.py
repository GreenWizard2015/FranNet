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

def showDiffusion(diff, img, N=14, process=lambda x: x, stepBy=25):
  steps = [img]
  for t in range(0, diff.noise_steps, stepBy):
    T = np.full(img.shape[:2], t)
    (xT, _, _) = diff.forward(img, T)
    steps.append(xT.numpy())
    continue

  for i in range(0, len(steps), N):
    items = steps[i:i+N]
    items = [cv2.resize(process(x)*255, (128, 128)) for x in items]
    cv2_imshow(np.concatenate(items, axis=1))
    continue
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