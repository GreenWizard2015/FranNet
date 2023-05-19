import tensorflow as tf

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

class CBasicInterpolantSampler:
  def __init__(self, interpolant, algorithm):
    self._interpolant = interpolant
    self._algorithm = algorithm
    return
  
  @property
  def interpolant(self): return self._interpolant
  
  @tf.function
  def sample(self, value, model, **kwargs):
    kwargs = dict(**kwargs, interpolant=self._interpolant) # add interpolant to kwargs
    step = self._algorithm.firstStep(value=value, **kwargs)
    while tf.reduce_any(step.active):
      x_hat = self._algorithm.inference(model=model, step=step, value=value, **kwargs)
      # solve
      solution = self._algorithm.solve(x_hat=x_hat, step=step, value=value, **kwargs)
      # make next step
      step = self._algorithm.nextStep(x_hat=x_hat, step=step, solution=solution, value=value, **kwargs)
      # update value
      tf.assert_equal(tf.shape(value), tf.shape(solution.value))
      value = solution.value
      continue
    
    return value
# End of CBasicInterpolantSampler