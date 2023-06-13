from NN.restorators.CNoiseProvider import noise_provider_from_config

class CDiffusionModelProxy:
  def __init__(self, model):
    self._model = model
    return
  
  def __call__(self, stochasticity, K, clipping, projectNoise, noiseStddev, **kwargs):
    assert isinstance(noiseStddev, str), f'Invalid noiseStddev: {noiseStddev}'
    reverseArgs = dict(
      stochasticity=float(stochasticity),
      projectNoise=True if projectNoise else False, # convert to bool
      clipping=dict(min=-1.0, max=1.0) if clipping else None,
      stepsConfig=dict(name='uniform', K=K),
      noiseProvider=noise_provider_from_config(noiseStddev)
    )
    return self._model(reverseArgs=reverseArgs, **kwargs)
  
  @property
  def kind(self): return self._model.kind
  
  @property
  def name(self): return self._model.name
# End of CDiffusionModelProxy
