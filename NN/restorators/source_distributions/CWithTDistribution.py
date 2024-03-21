import tensorflow as tf
import tensorflow_probability as tfp
from NN.restorators.source_distributions.ISourceDistribution import ISourceDistribution
import numpy as np

class CWithTDistribution(ISourceDistribution):
  def __init__(self, distribution, TFunction):
    self._distribution = distribution
    self._TFunction = TFunction
    return
  
  def sampleFor(self, x0):
    B = tf.shape(x0)[0]
    res = self._distribution.sampleFor(x0)
    return {
      'xT': res['xT'],
      'T': self._TFunction(B),
    }
  
  def initialValueFor(self, shape_or_values):
    return self._distribution.initialValueFor(shape_or_values)
  
def adjustedLN(mu, sigma, steps):
  def F(B):
    # oversample the first few steps, as they are more important
    index = tf.linspace(0., 1., 1 + steps)[1:]
    pdfA = tf.exp(-tf.square(tf.math.log(index) - mu) / (2. * sigma * sigma))
    pdfB = index * (sigma * np.sqrt(2. * np.pi))
    pdf = pdfA / pdfB

    weights = tf.nn.softmax(pdf, axis=-1)[None]
    res = tf.random.categorical(tf.math.log(weights), B)
    res = tf.cast(res, tf.float32)
    
    res = tf.reshape(res, (B, 1))
    return res / steps
  return F

def TFunction(config):
  assert isinstance(config, dict), 'config for TFunction must be a dict'
  assert 'name' in config, 'config for TFunction must have a name'
  name = config['name'].lower()
  if 'log normal' == name:
    return adjustedLN(
      mu=config['mu'],
      sigma=config['sigma'],
      steps=config['steps'],
    )
  raise ValueError('Unknown TFunction')

# for testing
if __name__ == "__main__":
  import numpy as np
  import matplotlib.pyplot as plt
  x0 = adjustedLN(-1, .5, 100)(100000)

  x0 = x0.numpy().flatten()
  # x0 = x0[x0 > 100]
  print(x0.shape)
  plt.hist(x0, bins=100)
  plt.show()
  pass