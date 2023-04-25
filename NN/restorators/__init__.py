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
