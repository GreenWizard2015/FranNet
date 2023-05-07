import tensorflow as tf
from NN.restorators.diffusion.diffusion_samplers import sampler_from_config

DDPM_CONFIG = {
  'name': 'ddpm',
  'noise stddev': 'zero',
}
def make_sampler(config):
  return sampler_from_config(dict(DDPM_CONFIG, **config))

def test_steps():
  sampler = make_sampler({})
  steps = sampler._stepsSequence(10, 0)
  tf.assert_equal(steps, [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
  return
