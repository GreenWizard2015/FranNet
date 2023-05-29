from ..CNoiseProvider import noise_provider_from_config
from .CDDPMSampler import CDDPMSampler
from .CDDIMSampler import CDDIMSampler

def sampler_from_config(config):
  assert isinstance(config, dict), 'Sampler config must be a dictionary'
  name = config['name'].lower()
  if 'ddpm' == name:
    return CDDPMSampler(
      noise_provider=noise_provider_from_config(config['noise stddev']),
      clipping=config.get('clipping', None),
    )
  
  if 'ddim' == name:
    return CDDIMSampler(
      stochasticity=config['stochasticity'],
      noise_provider=noise_provider_from_config(config['noise stddev']),
      steps=config['steps skip type'],
      clipping=config.get('clipping', None),
    )
  
  raise ValueError('Unknown sampler: %s' % config)