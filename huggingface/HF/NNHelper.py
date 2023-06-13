import os, glob
from Utils.utils import JSONHelper, load_config
from HF.NN.CDiffusionModelProxy import CDiffusionModelProxy
from HF.NN.CHuggingFaceBasicModel import CHuggingFaceBasicModel
from HF.NN.CARDirectionModelProxy import CARDirectionModelProxy
from HF.NN.CWithAblations import CWithAblations

def inference_from_config(configs):
  if 'huggingface' in configs:
    model = CHuggingFaceBasicModel(configs)
    model = CWithAblations(model) # add ablation parameters support
    
    if 'diffusion' == model.kind: model = CDiffusionModelProxy(model)
    if 'autoregressive direction' == model.kind: model = CARDirectionModelProxy(model)
    return model
  
  raise NotImplementedError('Something went wrong with the configs')

def modelsFrom(folder):
  models = []
  for modelFullPath in glob.glob(os.path.join(folder, '**', '*.json'), recursive=True):
    HFCard = JSONHelper.load(modelFullPath).get('huggingface', {})
    name = HFCard.get('name', None)
    if name is None: continue

    fullConfig = load_config(modelFullPath, folder=folder)
    models.append( inference_from_config(fullConfig) )
    continue
  return models
