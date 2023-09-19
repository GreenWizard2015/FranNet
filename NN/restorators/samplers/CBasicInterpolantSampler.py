import tensorflow as tf
from .ISamplingAlgorithm import ISamplingAlgorithm
from NN.utils import is_namedtuple

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
    # wrap algorithm with hook, if provided
    algorithm = kwargs.get('algorithmInterceptor', lambda x: x)( self._algorithm )
    assert isinstance(algorithm, ISamplingAlgorithm), f'Algorithm must be an instance of ISamplingAlgorithm, but got {type(algorithm)}'

    # perform sampling
    step = algorithm.firstStep(value=value, iteration=0, **kwargs)
    # CFakeObject is a namedtuple, so we need to check for it
    if isinstance(step, tuple) and not is_namedtuple(step):
      step, kwargs = step # store some data in kwargs, because TF a bit stupid

    iteration = tf.constant(1, dtype=tf.int32) # first step is already done
    while tf.reduce_any(step.active):
      KWArgs = dict(
        value=value, step=step, iteration=iteration,
        **kwargs
      ) # for simplicity
      # inference
      x_hat = algorithm.inference(model=model, **KWArgs)
      # solve
      solution = algorithm.solve(x_hat=x_hat, **KWArgs)
      # make next step
      step = algorithm.nextStep(x_hat=x_hat, solution=solution, **KWArgs)
      # update value
      tf.assert_equal(tf.shape(value), tf.shape(solution.value))
      value = solution.value
      iteration += 1
      continue
    
    return value
# End of CBasicInterpolantSampler