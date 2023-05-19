from .CDDIMInterpolantSampler import CDDIMInterpolantSampler
from ..interpolants import interpolant_from_config
from ..CNoiseProvider import noise_provider_from_config
from .CARSampler import autoregressive_sampler_from_config

def sampler_from_config(config):
  kind = config['name'].lower()
  if 'ddim' == kind:
    from ..diffusion.diffusion_schedulers import schedule_from_config

    interpolantConfig = config.get('interpolant', dict(name='diffusion'))
    return CDDIMInterpolantSampler(
      interpolant=interpolant_from_config(interpolantConfig),
      stochasticity=config['stochasticity'],
      noiseProvider=noise_provider_from_config(config['noise stddev']),
      schedule=schedule_from_config(config['schedule']),
      steps=config['steps skip type'],
      clipping=config.get('clipping', None),
    )
  
  if 'autoregressive' == kind:
    return autoregressive_sampler_from_config(config)
  
  raise ValueError(f'Unknown sampler kind: {kind}')