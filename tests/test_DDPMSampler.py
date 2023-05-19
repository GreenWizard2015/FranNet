import tensorflow as tf
from NN.restorators.diffusion.diffusion_samplers import sampler_from_config

DDPM_CONFIG = {
  'name': 'ddpm',
  'noise stddev': 'zero',
}
def make_sampler(config):
  return sampler_from_config(dict(DDPM_CONFIG, **config))