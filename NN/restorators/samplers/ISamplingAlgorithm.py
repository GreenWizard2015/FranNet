
class ISamplingAlgorithm:
  def firstStep(self, **kwargs):
    raise NotImplementedError()
  
  def nextStep(self, **kwargs):
    raise NotImplementedError()
  
  def inference(self, **kwargs):
    raise NotImplementedError()
  
  def solve(self, **kwargs):
    raise NotImplementedError()
# End of ISamplingAlgorithm