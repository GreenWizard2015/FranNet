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
  
  def train(self, x0, x1, T, xT=None):
    if xT is None:
      xT = self.interpolate(x0, x1, T)
    else:
      signal_rate, noise_rate = tf.sqrt(T), tf.sqrt(1.0 - T)
      x1 = (xT - (signal_rate * x0)) / noise_rate

    inputs = self.inference(xT=xT, T=T)
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
    # based on https://github.com/beresandras/clear-diffusion-keras/blob/master/model.py#L88-L90
    signal_rate, noise_rate = tf.sqrt(t), tf.sqrt(1.0 - t)
    x0 = xt * signal_rate - x_hat * noise_rate
    x1 = xt * noise_rate + x_hat * signal_rate
    return CFakeObject(x0=x0, x1=x1)
  
  def train(self, x0, x1, T, xT=None):
    if xT is None:
      xT = self.interpolate(x0, x1, T)
    else:
      signal_rate, noise_rate = tf.sqrt(T), tf.sqrt(1.0 - T)
      x1 = (xT - (signal_rate * x0)) / noise_rate
      tf.debugging.assert_near(xT, self.interpolate(x0, x1, T), atol=1e-6)

    inputs = self.inference(xT=xT, T=T)
    # calculate velocity
    # based on https://github.com/beresandras/clear-diffusion-keras/blob/master/model.py#L347
    signal_rate, noise_rate = tf.sqrt(T), tf.sqrt(1.0 - T)
    velocity = (signal_rate * x1) - (noise_rate * x0)
    return {
      'target': velocity, # predict velocity
      'x0': x0,
      'x1': x1,
      **inputs,
    }
# end CDiffusionInterpolantV