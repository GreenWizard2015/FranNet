import os, glob, shutil
from Utils.utils import JSONHelper, load_config
from Utils.WandBUtils import CWBRun
from NN import model_from_config
from HF.Utils import hashable_lru
from HF.NN.CDiffusionModelProxy import CDiffusionModelProxy

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

class CNNModel:
  def __init__(self, configs):
    self._configs = configs
    return
  
  def __call__(self, images, **kwargs):
    model = NN_From(self._configs)
    upscaled = model(images, **kwargs)
    return upscaled
  
  def _card(self):
    return self._configs['huggingface']
  
  @property
  def kind(self): return self._card()['kind']
  
  @property
  def name(self): return self._card()['name']
# End of CNNModel

def modelsFrom(folder):
  models = {}
  for modelFullPath in glob.glob(os.path.join(folder, '**', '*.json'), recursive=True):
    HFCard = JSONHelper.load(modelFullPath).get('huggingface', {})
    name = HFCard.get('name', None)
    if name is None: continue

    fullConfig = load_config(modelFullPath, folder=folder)
    model = CNNModel(fullConfig)
    if 'diffusion' == model.kind:
      model = CDiffusionModelProxy(model)

    models[model.name] = model
    continue
  return models
