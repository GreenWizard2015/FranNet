from Utils.utils import upgrade_configs_structure
import pytest
import json

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

def test_upgrade_renderer():
  config = {
    "model": {
      "decoder": "decoder",
      "restorator": "restorator",
      "renderer": {
        "position encoding": "position encoding",
        "time encoding": "time encoding",
        "batch_size": 32
      }
    }
  }
  config = upgrade_configs_structure(config)
  expected = {
    "model": {
      "renderer": {
        "batch_size": 32,
        "restoration model": {
          "name": "basic",
          "restorator": "restorator",
          "decoder": "decoder",
          "position encoding": "position encoding",
          "time encoding": "time encoding",
        }
      }
    }
  }
  assert json.dumps(config, indent=2, sort_keys=True) == json.dumps(expected, indent=2, sort_keys=True)
  return