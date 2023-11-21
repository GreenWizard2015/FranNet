class IRestorationProcess:
  def __init__(self, channels, **kwargs):
    super().__init__(**kwargs)
    self._channels = channels
    return
  
  def forward(self, x0):
    raise NotImplementedError()
  
  def reverse(self, value, denoiser, modelT=None, **kwargs):
    '''
    This function implements the reverse restorator process.

    value: (B, ...) is the initial value. Typically, this is a just a noise sample.
           Can be a tuple with batch shape (B, ...)
    denoiser: callable (x, t) -> x. Denoiser function or model, which takes a value and a continuous time t and returns a denoised value.
    modelT: callable (t) -> t. Model for the time parameter. If None, then the time parameter is assumed to be a continuous value in [0, 1].

    returns: (B, ...) denoised value
    '''
    raise NotImplementedError()
  
  def calculate_loss(self, x_hat, predicted, **kwargs):
    raise NotImplementedError()
  
  def train_step(self, x0, model, **kwargs):
    x_hat = self.forward(x0)
    values = model(T=x_hat['T'], V=x_hat['xT'])
    return self.calculate_loss(x_hat, values, **kwargs)
    
  @property
  def channels(self):
    return self._channels