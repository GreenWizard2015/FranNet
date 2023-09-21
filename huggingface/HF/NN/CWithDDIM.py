from NN.restorators.CNoiseProvider import noise_provider_from_config

class CWithDDIM:
  def __init__(self, model):
    self._model = model
    return
  
  def __call__(self, 
    DDIM_stochasticity, DDIM_K, DDIM_clipping, DDIM_projectNoise, DDIM_noiseStddev,
    reverseArgs={}, **kwargs
  ):
    assert isinstance(DDIM_noiseStddev, str), f'Invalid noiseStddev: {DDIM_noiseStddev}'
    reverseArgs = dict(
      **reverseArgs,
      stochasticity=float(DDIM_stochasticity),
      projectNoise=True if DDIM_projectNoise else False, # convert to bool
      clipping=dict(min=-1.0, max=1.0) if DDIM_clipping else None,
      stepsConfig=dict(name='uniform', K=DDIM_K),
      noiseProvider=noise_provider_from_config(DDIM_noiseStddev)
    )
    return self._model(reverseArgs=reverseArgs, **kwargs)
  
  @property
  def kind(self): return self._model.kind
  
  @property
  def name(self): return self._model.name
# End of CWithDDIM
