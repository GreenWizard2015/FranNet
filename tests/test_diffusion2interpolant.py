from NN.restorators import replace_diffusion_restorator_by_interpolant
import json

def _common_checks(newConfig, channels=3):
  assert isinstance(newConfig, dict), 'Must return a dictionary'
  assert 'autoregressive' == newConfig['name'], 'Must return an autoregressive restorator'
  assert 'source distribution' in newConfig, 'Must have a source distribution'
  assert 'sampler' in newConfig, 'Must have a sampler'
  sampler = newConfig['sampler']
  assert 'schedule' in sampler, 'Must have a schedule'
  # validate the sampler
  assert 'DDIM' == sampler['name'], 'Must have a DDIM sampler'
  assert 'diffusion' == sampler['interpolant']['name'], 'Must have a diffusion interpolant'

  assert channels == newConfig['channels'], 'Must have the same number of channels'
  assert 'steps skip type' in sampler, 'Must have a steps skip type'
  return

def test_not_diffusion():
  config = ''' {"name": "autoregressive"} '''
  newConfig = replace_diffusion_restorator_by_interpolant(json.loads(config))
  assert isinstance(newConfig, dict), 'Must return a dictionary'
  assert 'autoregressive' == newConfig['name'], 'Must return an autoregressive restorator'
  return

def test_DDPM_basic():
  config = '''
{
  "channels": 3,
  "name": "diffusion",
  "kind": "DDPM",
  "source distribution": "dummy distribution",
  "sampler": {
    "name": "DDPM",
    "noise stddev": "nstd",
    "clipping": "dummy clipping"
  },
  "schedule": "dummy schedule"
}
  '''
  newConfig = replace_diffusion_restorator_by_interpolant(json.loads(config))
  _common_checks(newConfig)

  sampler = newConfig['sampler']
  assert 1.0 == sampler['stochasticity'], 'Must set stochasticity to 1.0'
  assert 'nstd' == sampler['noise provider'], 'Must copy the noise stddev and rename it to noise provider'
  assert 'dummy distribution' == newConfig['source distribution'], 'Must copy the source distribution'
  assert 'dummy schedule' == sampler['schedule'], 'Must copy the schedule'
  assert 'dummy clipping' == sampler['clipping'], 'Must copy the clipping'

  # dummy 'steps skip type'
  assert 'uniform' == sampler['steps skip type']['name'], 'Must add a steps skip type'
  assert 1 == sampler['steps skip type']['K'], 'Must add a steps skip type'
  return

def test_DDIM():
  config = '''
{
  "channels": 3,
  "name": "diffusion",
  "kind": "DDIM",
  "source distribution": "dummy",
  "sampler": {
    "name": "DDIM",
    "noise stddev": "normal",
    "stochasticity": 0.5,
    "steps skip type": "skip type",
    "project noise": "project noise"
  },
  "schedule": "dummy"
}
  '''
  newConfig = replace_diffusion_restorator_by_interpolant(json.loads(config))
  _common_checks(newConfig)
  assert 0.5 == newConfig['sampler']['stochasticity'], 'Must copy the stochasticity'
  assert 'project noise' == newConfig['sampler']['project noise'], 'Must copy the project noise'
  return