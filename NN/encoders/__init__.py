from .Encoder import CEncoder, createEncoderHead
from .extractors import extractor_from_config

def encoder_from_config(config):
  if 'basic' == config['name']:
    imgWidth = config['image size']
    latentDim = config['latent dimension']
    headConfig = config['head']
    head = lambda name: createEncoderHead(
      imgWidth=imgWidth,
      channels=config['channels'],
      downsampleSteps=headConfig['downsample steps'],
      latentDim=latentDim,
      localContext=headConfig['local context'],
      globalContext=headConfig['global context'],
      name=name
    )

    mixer = config['contexts mixer']
    return CEncoder(
      imgWidth=imgWidth,
      channels=config['channels'],
      head=head,
      extractor=extractor_from_config(mixer['extractor'], latentDim)
    )
  
  raise ValueError(f"Unknown encoder name: {config['name']}")
 