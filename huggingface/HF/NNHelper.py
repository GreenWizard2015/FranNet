import os, glob, copy
from Utils.utils import JSONHelper, load_config
from HF.NN.CWithDDIM import CWithDDIM
from HF.NN.CHuggingFaceBasicModel import CHuggingFaceBasicModel
from HF.NN.CARDirectionModelProxy import CARDirectionModelProxy
from HF.NN.CWithAblations import CWithAblations
from HF.NN.CInterpolantVisualization import CInterpolantVisualization
from NN.restorators import replace_diffusion_restorator_by_interpolant

def _single_inference_from_config(configs):
  if 'huggingface' in configs:
    model = CHuggingFaceBasicModel(configs)
    model = CWithAblations(model) # add ablation parameters support
    
    if 'diffusion' in model.kind: model = CWithDDIM(model)
    if 'autoregressive direction' == model.kind: model = CARDirectionModelProxy(model)
    
    if ('autoregressive' in model.kind) or ('interpolant' in model.kind):
      model = CInterpolantVisualization(model)
    return model
  raise NotImplementedError('Something went wrong with the configs')

def inference_from_config(configs):
  model = _single_inference_from_config(configs)
  yield model
  # convert diffusion models also to interpolant/autoregressive model
  if 'diffusion' == model.kind:
    # convert restorator
    fullConfig = copy.deepcopy(configs)
    fullConfig['model']['restorator'] = replace_diffusion_restorator_by_interpolant(
      fullConfig['model']['restorator']
    )
    # update model name and kind
    oldName = fullConfig['huggingface']['name']
    fullConfig['huggingface']['name'] = f'{oldName} (interpolant)'
    fullConfig['huggingface'].pop('kind', None)
    yield _single_inference_from_config(fullConfig)
  return

def modelsFrom(folder):
  models = []
  for modelFullPath in glob.glob(os.path.join(folder, '**', '*.json'), recursive=True):
    HFCard = JSONHelper.load(modelFullPath).get('huggingface', {})
    name = HFCard.get('name', None)
    if name is None: continue

    fullConfig = load_config(modelFullPath, folder=folder)
    for model in inference_from_config(fullConfig):
      models.append( model )
    continue
  return models
