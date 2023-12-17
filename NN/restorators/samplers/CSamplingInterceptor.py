from .ISamplingAlgorithm import ISamplingAlgorithm

class CSamplingInterceptor(ISamplingAlgorithm):
  def __init__(self, watcher, algorithm):
    self._watcher = watcher
    self._algorithm = algorithm
    return
  
  def firstStep(self, **kwargs):
    res = self._algorithm.firstStep(**kwargs)
    self._watcher._onStart(value=kwargs['value'], kwargs=kwargs)
    return res
  
  def nextStep(self, **kwargs):
    self._watcher._onNextStep(iteration=kwargs['iteration'], kwargs=kwargs)
    res = self._algorithm.nextStep(**kwargs)
    return res
  
  def inference(self, **kwargs):
    return self._algorithm.inference(**kwargs)
  
  def solve(self, **kwargs):
    return self._algorithm.solve(**kwargs)
# End of CSamplingInterceptor