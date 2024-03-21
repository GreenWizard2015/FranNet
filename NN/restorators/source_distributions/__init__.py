from .CNormalSourceDistribution import CNormalSourceDistribution
from .CUniformSourceDistribution import CUniformSourceDistribution
from .CSwitchSourceDistribution import CSwitchSourceDistribution
from .CResampledNormalSourceDistribution import CResampledNormalSourceDistribution
from .CWithTDistribution import CWithTDistribution, TFunction

def source_distribution_from_config(config):
  assert isinstance(config, dict), 'config for source distribution must be a dict'
  assert 'name' in config, 'config for source distribution must have a name'
  name = config['name'].lower()

  if 't distribution' == name:
    return CWithTDistribution(
      distribution=source_distribution_from_config(config['distribution']),
      TFunction=TFunction(config['TFunction']),
    )
  
  if 'normal' == name:
    return CNormalSourceDistribution(
      mean=config['mean'],
      stddev=config['stddev'],
    )
  
  if 'normal resampled' == name:
    return CResampledNormalSourceDistribution(
      mean=config['mean'],
      stddev=config['stddev'],
      fraction=config['fraction'],
    )
  
  # They all are uniform distributions
  if name in ['halton', 'sobol', 'uniform']:
    return CUniformSourceDistribution(
      min=config['min'],
      max=config['max'],
      distribution=name,
    )
  
  if 'switch' == name:
    return CSwitchSourceDistribution(
      sample_for=source_distribution_from_config(config['training']),
      initial_value_for=source_distribution_from_config(config['inference']),
    )
  
  raise ValueError('Unknown distribution')
