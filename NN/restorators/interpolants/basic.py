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
    # x_hat is the direction towards x0
    return CFakeObject(x0=xt + x_hat, x1=xt)
  
  def train(self, x0, x1, T, xT=None):
    xt = x1 if xT is None else xT
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
  
  def train(self, x0, x1, T, xT=None):
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