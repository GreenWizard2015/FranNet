import tensorflow as tf
from Utils.utils import CFakeObject

class CProcessStepsDecayed:
  def __init__(self, start, end, steps, decay):
    self._start = start
    self._end = end
    self._steps = steps
    self._decay = decay
    return
  
  def _at(self, step, **kwargs):
    step = tf.cast(step, tf.float32)
    current = tf.pow(self._decay, step) * self._start
    current = tf.clip_by_value(current, self._end, self._start)
    return current
  
  def at(self, step, **kwargs):
    current = self._at(step, **kwargs)
    prevT = self._at(step - 1, **kwargs)
    return CFakeObject(
      variance=prevT,
      T=current,
      prevT=prevT,
    )
  
  @property
  def limit(self): return self._steps
# End of CProcessStepsDecayed