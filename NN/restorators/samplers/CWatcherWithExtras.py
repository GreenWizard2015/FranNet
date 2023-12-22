from .CSamplingInterceptor import CSamplingInterceptor
from .ISamplerWatcher import ISamplerWatcher

class CWatcherWithExtras(ISamplerWatcher):
  def __init__(self, watcher, converter, residuals):
    self._watcher = watcher
    self._converter = converter
    self._residuals = residuals
    return

  @property
  def iteration(self):
    return self._watcher.iteration
    
  def interceptor(self):
    return lambda algorithm: CSamplingInterceptor(watcher=self, algorithm=algorithm)
  
  def _onStart(self, value, kwargs):
    return self._watcher._onStart(value=value, kwargs=kwargs)
  
  def _onNextStep(self, iteration, kwargs):
    return self._watcher._onNextStep(iteration=iteration, kwargs=kwargs)
  
  def _convert(self, value):
    if not(self._converter is None):
      value = self._converter.convertBack(value) # convert to RGB

    if not(self._residuals is None):
      value = value + self._residuals

    return value
  
  def tracked(self, name):
    res = self._watcher.tracked(name)
    if name in ['value', 'x0', 'x1']:
      res = self._convert(res)
    return res
# End of CWatcherWithExtras