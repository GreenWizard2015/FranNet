from NN.restorators.CNoiseProvider import noise_provider_from_config
from NN.restorators.samplers.steps_schedule import CProcessStepsDecayed

class CARDirectionModelProxy:
  def __init__(self, model):
    self._model = model
    return
  
  def __call__(self, threshold, start, end, steps, decay, noiseStddev, convergeThreshold, **kwargs):
    assert isinstance(noiseStddev, str), f'Invalid noiseStddev: {noiseStddev}'
    # convert arguments to floats, because gradio passes them sometimes as ints
    threshold = float(threshold)
    start = float(start)
    end = float(end)
    steps = int(steps)
    decay = float(decay)
    convergeThreshold = float(convergeThreshold)

    reverseArgs = dict(
      threshold=threshold,
      convergeThreshold=convergeThreshold,
      steps=CProcessStepsDecayed(start, end, steps, decay),
      noiseProvider=noise_provider_from_config(noiseStddev)
    )
    return self._model(reverseArgs=reverseArgs, **kwargs)
  
  @property
  def kind(self): return self._model.kind
  
  @property
  def name(self): return self._model.name
# End of CDiffusionModelProxy
