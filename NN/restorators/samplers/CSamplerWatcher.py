import tensorflow as tf
from NN.utils import is_namedtuple
from .CSamplingInterceptor import CSamplingInterceptor
from .ISamplerWatcher import ISamplerWatcher

class CSamplerWatcher(ISamplerWatcher):
  def __init__(self, steps, tracked, indices=None):
    super().__init__()
    self._indices = tf.constant(indices, dtype=tf.int32) if not(indices is None) else None
    self._tracked = {}
    prefix = [steps]
    if not(self._indices is None):
      prefix = [steps, tf.size(self._indices)]

    for name, shape in tracked.items():
      shp = prefix + list(shape)
      self._tracked[name] = tf.Variable(tf.zeros(shp), trainable=False)
      continue

    if 'value' in self._tracked: # value has steps + 1 shape, so we need extra variable
      shp = prefix + list(tracked['value'])
      self._initialValue = tf.Variable(tf.zeros(shp[1:]), trainable=False)
      pass

    self._iteration = tf.Variable(0, trainable=False)
    return

  @property
  def iteration(self):
    return self._iteration.read_value() + 1
    
  def interceptor(self):
    return lambda algorithm: CSamplingInterceptor(watcher=self, algorithm=algorithm)
  
  def tracked(self, name):
    res = self._tracked.get(name, None)
    if res is None: return None
    if 'value' == name: # add initial value
      res = tf.concat([self._initialValue[None], res], axis=0)
    return res
  
  def _updateTracked(self, name, value, mask=None):
    tracked = self._tracked.get(name, None)
    if tracked is None: return
    value = self._withIndices(value)
    
    iteration = self._iteration
    if (mask is None) or ('value' == name): # 'value' is always unmasked
      tracked[iteration].assign(value)
      return
    
    mask = self._withIndices(mask)
    prev = tracked[iteration - 1]
    # expand mask to match the value shape by copying values from the previous iteration
    indices = tf.where(mask)
    value = tf.tensor_scatter_nd_update(prev, indices, value)
    tf.assert_equal(tf.shape(prev), tf.shape(value), 'Must be the same shape')
    tracked[iteration].assign(value)
    return
  
  def _onNextStep(self, iteration, kwargs):
    self._iteration.assign(iteration)
    # track also solution
    solution = kwargs['solution']
    assert is_namedtuple(solution), 'Solution must be a namedtuple'
    step = kwargs['step']
    assert is_namedtuple(step), 'Step must be a namedtuple'
    mask = step.mask if hasattr(step, 'mask') else None
    # iterate over all fields
    for name in solution._fields:
      self._updateTracked(name, getattr(solution, name), mask=mask)
      continue
    return
  
  def _onStart(self, value, kwargs):
    self._iteration.assign(0)
    if 'value' in self._tracked: # save initial value
      self._initialValue.assign( self._withIndices(value) )
    return
  
  def _withIndices(self, value):
    if self._indices is None: return value
    return tf.gather(value, self._indices, axis=0)
# End of CSamplerWatcher