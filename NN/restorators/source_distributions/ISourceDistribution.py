'''
Base class for source distributions.
sampleFor(x0) - returns a samples for the given x0
initialValueFor(shape_or_values) - returns an initial value for the given shape or values
'''
class ISourceDistribution:
  def sampleFor(self, x0):
    raise NotImplementedError('Not implemented yet')
  
  def initialValueFor(self, shape_or_values):
    raise NotImplementedError('Not implemented yet')
