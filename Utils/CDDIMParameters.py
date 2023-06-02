class CDDIMParameters:
  def __init__(self, stochasticity, stddev, K, clipping=None, projectNoise=False):
    self.stochasticity = stochasticity
    self.stddev = stddev
    self.K = K
    self.clipping = clipping
    self.projectNoise = projectNoise
    pass

  def toStats(self):
    return {
      'stochasticity': self.stochasticity,
      'noise stddev': self.stddev,
      'steps skip type': 'uniform',
      'K': self.K,
      'clipping': self.clipping is not None,
      'project noise': self.projectNoise,
    }
  
  def toModelConfig(self):
    return {
      'model': {
        'restorator': {
          'sampler': {
            'name': 'DDIM',
            'stochasticity': self.stochasticity,
            'noise stddev': self.stddev,
            'steps skip type': { 'name': 'uniform', 'K': self.K },
            'clipping': self.clipping,
            'project noise': self.projectNoise,
          }
        }
      }
    }
  
  @staticmethod
  def fromStats(stats):
    clipping = None
    if stats['clipping']:
      clipping = dict(min=-1.0, max=1.0)

    return CDDIMParameters(
      stochasticity=stats['stochasticity'],
      stddev=stats['noise stddev'],
      K=stats['K'],
      clipping=clipping,
      projectNoise=stats['project noise']
    )
# End of CDDIMParameters
  
def DDIMParameters(stochasticity, stddev, K, clipping=None, projectNoise=False):
  params = CDDIMParameters(stochasticity, stddev, K, clipping, projectNoise)
  return (
    params.toModelConfig(),
    params.toStats()
  )
