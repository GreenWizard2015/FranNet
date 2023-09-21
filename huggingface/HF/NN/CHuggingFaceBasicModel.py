import os, shutil
from Utils.WandBUtils import CWBRun
from NN import model_from_config
from HF.Utils import hashable_lru

# prevent downloading the same weights file multiple times
@hashable_lru(maxsize=None)
def WBWeightsFile(configs):
  WBRun = CWBRun(configs['wandb'])
  tmpFile = f'tmp/{WBRun.id}.h5'
  if os.path.exists(tmpFile):
    return tmpFile

  weightsFile = configs.get('weights', None)
  if weightsFile is not None:
    weightsFile = next([f for f in WBRun.models() if weightsFile == f.name], None)
    assert weightsFile is not None, f'Invalid weights file: {weightsFile}'
  else:
    weightsFile = WBRun.bestModel
    
  weightsFile = weightsFile.pathTo()

  # copy weights file to "tmp/{run.id}.h5"
  os.makedirs(os.path.dirname(tmpFile), exist_ok=True)
  shutil.copyfile(weightsFile, tmpFile)
  return tmpFile

@hashable_lru(maxsize=100)
def NN_From(configs):
  model = model_from_config(configs['model'])
  model.load_weights(WBWeightsFile(configs['huggingface']))
  return model

def _autoregressive_kind(restorator):
  sampler = restorator['sampler']['name'].lower()
  interpolant = restorator['sampler']['interpolant'].get('name', '').lower()

  if 'autoregressive' == sampler:
    if 'direction' == interpolant: return 'autoregressive direction'
  
  if 'ddim' == sampler: return 'autoregressive diffusion'
  raise NotImplementedError(f'Unknown sampler: {sampler}')

class CHuggingFaceBasicModel:
  def __init__(self, configs):
    self._configs = configs
    return
  
  def __call__(self, images, targetResolution, raw=None, **kwargs):
    targetResolution = int(targetResolution)
    assert 128 <= targetResolution <= 1024, f'Invalid target resolution: {targetResolution}'
    
    model = NN_From(self._configs)
    upscaled = model(images, size=targetResolution, **kwargs)
    return upscaled
  
  def _card(self):
    return self._configs['huggingface']
  
  @property
  def kind(self):
    card = self._card()
    if 'kind' in card: return card['kind']
    # get kind from model definition
    restorator = self._configs['model']['restorator']
    name = restorator['name']
    if 'autoregressive' == name:
      return _autoregressive_kind(restorator)
    
    return name
  
  @property
  def name(self): return self._card()['name']
# End of CHuggingFaceBasicModel