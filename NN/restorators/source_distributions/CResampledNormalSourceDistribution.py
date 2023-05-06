import tensorflow as tf
import tensorflow_probability as tfp
from .ISourceDistribution import ISourceDistribution

'''
Performs importance sampling from a normal distribution to find more rare samples.
On average, this algorithm increases the number of samples beyond 3sigma by 1.0 / fraction times.
'''
class CResampledNormalSourceDistribution(ISourceDistribution):
  def __init__(self, mean, stddev, fraction):
    super().__init__()
    self._mean = mean
    self._stddev = stddev
    self._fraction = fraction
    return
  
  def sampleFor(self, x0):
    B = tf.shape(x0)[0]
    return {
      'xT': self.initialValueFor(x0),
      'T': tf.random.uniform((B, 1), 0.0, 1.0),
    }
  
  @tf.function
  def initialValueFor(self, shape_or_values):
    if not isinstance(shape_or_values, tuple):
      shape_or_values = tf.shape(shape_or_values)
      tf.assert_equal(tf.shape(shape_or_values), (2, ))
    
    B, N = shape_or_values[0], shape_or_values[1]
    # make multivariate normal distribution
    distribution = tfp.distributions.MultivariateNormalDiag(
      loc=tf.fill((N,), self._mean),
      scale_diag=tf.fill((N,), self._stddev),
    )

    def _sample(M):
      res = distribution.sample(M)
      prob = distribution.log_prob(res)
      tf.assert_equal(tf.shape(prob), (M, ))
      tf.assert_equal(tf.shape(res), (M, N))
      return res, prob
    ###
    res, probs = _sample(B)
    fraction = self._fraction
    f = tf.cast(B, tf.float32) * fraction
    M = tf.cast(tf.math.ceil(f), tf.int32) + 1
    while 0 < (B - M):
      # sample from the remaining samples
      res2, probs2 = _sample(B - M)
      # preserve M samples
      res2 = tf.concat([res[:M], res2], axis=0)
      probs2 = tf.concat([probs[:M], probs2], axis=0)
      # replace the samples with the higher probability
      better = probs2 < probs
      res = tf.where(better[..., None], res2, res)
      probs = tf.where(better, probs2, probs)

      f = tf.cast(B - M, tf.float32) * fraction
      M = M + tf.cast(tf.math.ceil(f), tf.int32) + 1
      continue

    return res
# End of CResampledNormalSourceDistribution

if __name__ == '__main__':
  import matplotlib.pyplot as plt
  sd = CResampledNormalSourceDistribution(0.0, 1.0, 0.1)
  N = 10000
  limit = 3.0
  x0 = sd.initialValueFor((N, 2)).numpy()
  # remove all samples with distance < 3
  x0 = x0[limit < tf.norm(x0, axis=1)]
  print(x0.shape)
  
  f, ax = plt.subplots(1, 2, sharex=True, sharey=True)
  ax[0].hist2d(x0[:, 0], x0[:, 1], bins=300)

  normal = tf.random.normal((N, 2)).numpy()
  normal = normal[limit < tf.norm(normal, axis=1)]
  print(normal.shape)
  ax[1].hist2d(normal[:, 0], normal[:, 1], bins=300)
  
  d = 4
  plt.xlim(-d, d)
  plt.ylim(-d, d)
  plt.show()