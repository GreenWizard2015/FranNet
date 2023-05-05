from .CConvExtractor import conv_extractor_from_config
from .CInterpolateExtractor import interpolate_extractor_from_config
from .CCombinedExtractor import combined_extractor_from_config

def extractor_from_config(config, latentDim):
  name = config['name'].lower()
  if 'interpolate' == name:
    return interpolate_extractor_from_config(config, latentDim)
  
  if 'conv' == name:
    return conv_extractor_from_config(config, latentDim)
  
  # combined extractor
  if 'combined' == name:
    extractors = [extractor_from_config(m, latentDim) for m in config['extractors']]
    return combined_extractor_from_config(config, extractors)
  
  raise ValueError(f'Unknown extractor method: {name}')
