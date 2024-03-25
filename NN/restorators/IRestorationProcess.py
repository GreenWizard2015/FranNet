class IRestorationProcess:
  def __init__(self, channels, **kwargs):
    super().__init__(**kwargs)
    self._channels = channels
    return
  
  def forward(self, x0, xT=None):
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
  
  def train_step(self, x0, model, xT=None, **kwargs):
    x_hat = self.forward(x0=x0, xT=xT)
    values = model(T=x_hat['T'], V=x_hat['xT'])

    # we want calculate the loss WITH the residual
    totalLoss = self.calculate_loss(x_hat, values, **kwargs)
    return dict(
      loss=totalLoss,
      value=self.targets(x_hat, values)
    )
    
  def targets(self, x_hat, values):
    '''
    This function calculates the target values.
    x_hat: dictionary with the keys 'T' and 'xT'
    values: predicted values
    '''
    return values

  @property
  def channels(self):
    return self._channels