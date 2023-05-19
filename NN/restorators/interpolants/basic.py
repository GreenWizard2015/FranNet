import tensorflow as tf
from .IInterpolant import IInterpolant
from Utils.utils import CFakeObject

'''
Works by predicting direction towards x0 from xt.
'''
class CDirectionInterpolant(IInterpolant):
  def interpolate(self, x0, x1, t, **kwargs):
    # linear interpolation
    return x0 + (x1 - x0) * t
  
  def solve(self, x_hat, xt, t, **kwargs):
    return CFakeObject(x0=xt + x_hat, x1=x_hat)
  
  def train(self, x0, x1, T):
    xt = x1
    inputs = self.inference(xT=xt, T=T)
    return {
      'target': x0 - xt, # xt + x0 - xt == x0
      'x0': x0,
      'x1': x1,
      **inputs,
    }
  
  def inference(self, xT, T):
    B = tf.shape(xT)[0]
    return {
      'xT': xT,
      'T': tf.zeros((B, 1), dtype=tf.float32), # not used
    }
# End of CDirectionInterpolant

'''
Regardless of t and xt, always returns x0.
Same as single-pass restoration.
'''
class CConstantInterpolant(IInterpolant):
  def interpolate(self, x0, x1, t, **kwargs):
    tf.assert_equal(x0, x1)
    return x0
  
  def solve(self, x_hat, xt, t, **kwargs):
    return CFakeObject(x0=x_hat, x1=x_hat)
  
  def train(self, x0, x1, T):
    inputs = self.inference(xT=x1, T=T)
    return {
      'target': x0,
      'x0': x0,
      'x1': x1,
      **inputs,
    }
  
  def inference(self, xT, T):
    B = tf.shape(xT)[0]
    return {
      'xT': tf.zeros_like(xT), # not used
      'T': tf.zeros((B, 1), dtype=tf.float32), # not used
    }
# End of CConstantInterpolant

class CCommonInterpolantBase(IInterpolant):
  def __init__(self, targetT):
    self._targetT = targetT
    # small optimization
    self._getTarget = self.interpolate
    if targetT == 0.0: self._getTarget = lambda x0, x1, t: x0
    if targetT == 1.0: self._getTarget = lambda x0, x1, t: x1
    return
  
  def train(self, x0, x1, T):
    xt = self.interpolate(x0, x1, T)
    target = self._getTarget(x0, x1, T)
    inputs = self.inference(xT=xt, T=T)
    return {
      'target': target,
      'x0': x0,
      'x1': x1,
      **inputs,
    }
# End of CCommonInterpolantBase