
class ISamplerWatcher:
  def tracked(self, name):
    raise NotImplementedError('Must be implemented in derived class')
  
  def interceptor(self):
    # returns a function that takes an instance of ISamplingAlgorithm
    raise NotImplementedError('Must be implemented in derived class')
  
  def _onStart(self, value, kwargs):
    # called when the algorithm starts
    raise NotImplementedError('Must be implemented in derived class')
  
  def _onNextStep(self, iteration, kwargs):
    # called after each step
    raise NotImplementedError('Must be implemented in derived class')
# End of ISamplerWatcher