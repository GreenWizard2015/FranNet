
class ISamplingAlgorithm:
  def firstStep(self, **kwargs):
    raise NotImplementedError()
  
  def nextStep(self, **kwargs):
    raise NotImplementedError()
  
  def inference(self, **kwargs):
    raise NotImplementedError()
  
  def solve(self, **kwargs):
    raise NotImplementedError()
  
  def directSolve(self, x_hat, xt, T, interpolant):
    return x_hat
# End of ISamplingAlgorithm