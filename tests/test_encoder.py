import pytest
import numpy as np
import tensorflow as tf
from NN.encoders.Encoder import conv_block_params_from_config

def _check_basic(params):
  assert len(params) == 4
  for i in range(3):
    assert params[i]['channels'] == 3
    assert params[i]['kernel size'] == 3
    assert params[i]['activation'] == 'relu'
    continue
  assert params[3]['channels'] == 5
  assert params[3]['kernel size'] == 3
  assert params[3]['activation'] == 'relu'
  return

def test_list_ints():
  params = conv_block_params_from_config({
    'channels': 3,
    'conv before': [3, 3, 3],
    'channels last': 5,
  })
  _check_basic(params)
  return

def test_number_of_conv():
  params = conv_block_params_from_config({
    'channels': 3,
    'conv before': 3,
    'channels last': 5,
  })
  _check_basic(params)
  return