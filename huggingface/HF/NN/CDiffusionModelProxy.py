from NN.restorators.CNoiseProvider import noise_provider_from_config

class CDiffusionModelProxy:
  def __init__(self, model):
    self._model = model
    return
  
  def __call__(self, images, size, stochasticity, K, clipping, projectNoise, noiseStddev):
    assert isinstance(noiseStddev, str), f'Invalid noiseStddev: {noiseStddev}'
    reverseArgs = dict(
      stochasticity=stochasticity,
      projectNoise=True if projectNoise else False, # convert to bool
      clipping=dict(min=-1.0, max=1.0) if clipping else None,
      stepsConfig=dict(name='uniform', K=K),
      noiseProvider=noise_provider_from_config(noiseStddev)
    )
    return self._model(images, size=size, reverseArgs=reverseArgs)
  
  def _card(self):
    return self._model._configs['huggingface']
  
  @property
  def kind(self): return self._card()['kind']
  
  @property
  def name(self): return self._card()['name']
# End of CDiffusionModelProxy
