import tensorflow as tf
from .IInterpolant import IInterpolant
from Utils.utils import CFakeObject

'''
Implements the core of the diffusion process in form of an interpolant.
t is NOT the time parameter, rather it's alpha_hat.
'''

class CBaseDiffusionInterpolant(IInterpolant):
  def interpolate(self, x0, x1, t, t2=None):
    if t2 is None: t2 = 1.0 - t
    signal_rate, noise_rate = tf.sqrt(t), tf.sqrt(t2)
    return (signal_rate * x0) + (noise_rate * x1)
# End of CBaseDiffusionInterpolant

class CDiffusionInterpolant(CBaseDiffusionInterpolant):
  def solve(self, x_hat, xt, t):
    x1 = x_hat
    signal_rate, noise_rate = tf.sqrt(t), tf.sqrt(1.0 - t)
    x0 = (xt - (noise_rate * x1)) / signal_rate
    return CFakeObject(x0=x0, x1=x1)
  
  def train(self, x0, x1, T):
    xt = self.interpolate(x0, x1, T)
    inputs = self.inference(xT=xt, T=T)
    return {
      'target': x1, # predict noise
      'x0': x0,
      'x1': x1,
      **inputs,
    }
# end CDiffusionInterpolant

# Diffusion Interpolant with a V objective
class CDiffusionInterpolantV(CBaseDiffusionInterpolant):
  def solve(self, x_hat, xt, t):
    signal_rate, noise_rate = tf.sqrt(t), tf.sqrt(1.0 - t)
    x0 = xt * signal_rate - x_hat * noise_rate
    x1 = xt * noise_rate + x_hat * signal_rate
    return CFakeObject(x0=x0, x1=x1)
  
  def train(self, x0, x1, T):
    xt = self.interpolate(x0, x1, T)
    inputs = self.inference(xT=xt, T=T)
    target = self.interpolate(x0, x1, 1.0 - T)
    return {
      'target': target,
      'x0': x0,
      'x1': x1,
      **inputs,
    }
# end CDiffusionInterpolantV
