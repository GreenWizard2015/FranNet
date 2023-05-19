from .CDiffusionInterpolant import CDiffusionInterpolant, CDiffusionInterpolantV
from .basic import CDirectionInterpolant, CConstantInterpolant

def interpolant_from_config(config):
  kind = config['name'].lower()
  if 'diffusion' == kind:
    return CDiffusionInterpolant()
  
  if 'diffusion-v' == kind:
    return CDiffusionInterpolantV()
  
  if kind in ['direction', 'derivative']:
    return CDirectionInterpolant()
  
  if kind in ['constant', 'x0']: # predict x0 from xt
    return CConstantInterpolant()
  
  raise ValueError(f'Unknown interpolant kind: {kind}')