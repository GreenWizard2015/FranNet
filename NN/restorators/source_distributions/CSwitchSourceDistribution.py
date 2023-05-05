from .ISourceDistribution import ISourceDistribution

'''
This class is used to switch between different source distributions, so we can use normal distribution during inference and arbitrary distribution during training.
'''
class CSwitchSourceDistribution(ISourceDistribution):
  def __init__(self, sample_for, initial_value_for):
    super().__init__()
    self._sample_for = sample_for
    self._initial_value_for = initial_value_for
    return
  
  def sampleFor(self, x0):
    return self._sample_for.sampleFor(x0)
  
  def initialValueFor(self, x0):
    return self._initial_value_for.initialValueFor(x0)
# End of CSwitchSourceDistribution  