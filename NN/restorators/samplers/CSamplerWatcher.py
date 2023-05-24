import tensorflow as tf
from .ISamplingAlgorithm import ISamplingAlgorithm

class CSamplingInterceptor(ISamplingAlgorithm):
  def __init__(self, watcher, algorithm):
    self._watcher = watcher
    self._algorithm = algorithm
    return
  
  def firstStep(self, **kwargs):
    res = self._algorithm.firstStep(**kwargs)
    self._watcher._onStep(
      iteration=kwargs['iteration'],
      value=kwargs['value'],
      kwargs=kwargs
    )
    return res
  
  def nextStep(self, **kwargs):
    res = self._algorithm.nextStep(**kwargs)
    self._watcher._onStep(
      iteration=kwargs['iteration'],
      value=kwargs['solution'].value,
      kwargs=kwargs
    )
    return res
  
  def inference(self, **kwargs):
    res = self._algorithm.inference(**kwargs)
    return res
  
  def solve(self, **kwargs):
    res = self._algorithm.solve(**kwargs)
    return res
# End of CSamplingInterceptor

class CSamplerWatcher:
  def __init__(self, steps, tracked, indices=None):
    self._steps = steps
    self._indices = tf.constant(indices, dtype=tf.int32) if not(indices is None) else None
    self._tracked = {}
    prefix = [steps + 1]
    if not(self._indices is None):
      prefix = [steps + 1, tf.size(self._indices)]

    for name, shape in tracked.items():
      shp = prefix + list(shape)
      self._tracked[name] = tf.Variable(tf.zeros(shp), trainable=False)
      continue

    self._iteration = tf.Variable(0, trainable=False)
    return

  @property
  def iteration(self):
    return self._iteration.read_value() + 1
    
  def interceptor(self):
    return lambda algorithm: CSamplingInterceptor(watcher=self, algorithm=algorithm)
  
  def tracked(self, name):
    return self._tracked.get(name, None)
  
  def _updateTracked(self, name, value, iteration):
    tracked = self._tracked.get(name, None)
    if tracked is None: return
    if not(self._indices is None):
      value = tf.gather(value, self._indices, axis=0)
    tracked[iteration].assign(value)
    return
  
  def _onStep(self, iteration, value, kwargs):
    self._updateTracked('value', value, iteration=iteration)

    self._iteration.assign(iteration)
    return
# End of CSamplerWatcher