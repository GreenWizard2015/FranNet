
class IInterpolant:
  def interpolate(self, x0, x1, t):
    # if t 1.0, then return x1
    # if t 0.0, then return x0
    # else return mix of x0 and x1
    raise NotImplementedError()
  
  def solve(self, x_hat, xt, t):
    # find x0 such that interpolate(x0, x_hat, t) == xt
    raise NotImplementedError()
  
  def train(self, x0, x1, t, xT=None):
    # return a dictionary with following keys:
    # 'x' - input to the model
    # 'target' - value, that should be predicted
    # 'x0', 'x1' and 't' - same as input OR modified versions of them
    # Model inputs:
    # 'xT' - input x for the model, that should be used to predict 'target'
    # 'T' - input t for the model, that should be used to predict 'target'
    raise NotImplementedError()
  
  def inference(self, xT, T):
    # prepare input for the model
    return { 'xT': xT, 'T': T }