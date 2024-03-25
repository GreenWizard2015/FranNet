from .CRestorationModel import CRestorationModel
from .CSequentialRestorator import CSequentialRestorator
from .CRepeatedRestorator import CRepeatedRestorator
from NN.Decoder import decoder_from_config
from NN.encoding import encoding_from_config
from NN.restorators import restorator_from_config
from .Embeddings import CEncodedEmbeddings, CNumberEmbeddings
from NN.encoding import encoding_from_config

def embeddings_from_config(config, N):
  assert isinstance(config, dict)
  name = config['name'].lower()
  if 'embeddings' == name:
    return CNumberEmbeddings(N=N, D=config['dim'])
  
  name = config['name'].lower()
  if 'encoded' == name:
    return CEncodedEmbeddings(N=N, encoding=encoding_from_config(config['encoding']))
  raise ValueError(f"Unknown embeddings name: {config['name']}")

def restorationModel_from_config(config):
  name = config['name'].lower()
  if 'sequential' == name:
    restorators = [restorationModel_from_config(subConfig) for subConfig in config['restorators']]
    return CSequentialRestorator(restorators)
  
  if 'basic' == name:
    return CRestorationModel(
      decoder=decoder_from_config(config['decoder']),
      restorator=restorator_from_config(config['restorator']),
      posEncoder=encoding_from_config(config['position encoding']),
      timeEncoder=encoding_from_config(config['time encoding']),
    )
  
  if 'repeated' == name:
    return CRepeatedRestorator(
      restorator=restorationModel_from_config(config['restorator']),
      IDs=embeddings_from_config(config['IDs'], N=config['N']),
      N=config['N'],
    )
  raise ValueError(f"Unknown restoration model name: {config['name']}")
