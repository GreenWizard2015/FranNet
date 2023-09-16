from .CSingleStepRestoration import CSingleStepRestoration
from .CARProcess import autoregressive_restoration_from_config
from .diffusion import diffusion_from_config

def restorator_from_config(config):
  name = config['name']
  if 'autoregressive' == name:
    return autoregressive_restoration_from_config(config)
  
  if 'diffusion' == name:
    return diffusion_from_config(config)
  
  if 'single pass' == name:
    return CSingleStepRestoration(channels=config['channels'])
  
  raise ValueError(f"Unknown restorator name: {name}")

def replace_diffusion_restorator_by_interpolant(config):
  name = config['name']
  if 'diffusion' != name: return config

  origSampler = config['sampler']
  samplerParams = {
    'name': 'DDIM',
    'interpolant': dict(name='diffusion'),
    'noise provider': origSampler['noise stddev'],
    'schedule': config['schedule'],

    'stochasticity': origSampler.get('stochasticity', 1.0),
    'clipping': origSampler.get('clipping', None),
    'steps skip type': origSampler.get('steps skip type', dict(name='uniform', K=1)),
    'project noise': origSampler.get('project noise', False),
  }

  return {
    'channels': config['channels'],
    'name': 'autoregressive',
    'sampler': samplerParams,
    'source distribution': config['source distribution'],
  }