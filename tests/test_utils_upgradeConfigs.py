from Utils.utils import upgrade_configs_structure
import pytest

def default_config():
  return {
    "model": {
      "nerf": {
        "samplesN": 256,
        "training sampler": "structured"
      }
    },
    "dataset": {
      "test": {},
      "train": {},
    },
  }

def test_convert_samplesN():
  config = upgrade_configs_structure(default_config())
  assert 'samplesN' not in config['model']['nerf']
  train = config['dataset']['train']
  assert train['subsample']['N'] == 256
  return

def test_convert_sampler():
  config = upgrade_configs_structure(default_config())
  assert 'training sampler' not in config['model']['nerf']
  train = config['dataset']['train']
  assert train['subsample']['sampling'] == 'structured'
  return