import re

class CWithAblations:
  def __init__(self, model):
    self._model = model
    return
  
  def __call__(self, randomizePositions, encoderContext, encoderIntermediate, reverseArgs={}, **kwargs):
    encoderContextMappping = {
      'Both': (False, False),
      'Only local': (False, True),
      'Only global': (True, False),
    }
    noLocalContext, noGlobalContext = encoderContextMappping[encoderContext]

    # extract numbers from intermediate context names
    pattern = re.compile(r'(\d+)')
    intermediateContextsID = [int(pattern.search(name).group(1)) for name in encoderIntermediate]
    # populate dictionary with intermediate context names 'no intermediate {}'
    intermediateContexts = {
      ('no intermediate %d' % i): (i in intermediateContextsID)
      for i in range(1, 12)
    }

    reverseArgs = dict(
      **reverseArgs,
      decoder={ # add ablation parameters to decoder
        'randomize positions': bool(randomizePositions),
      },
      encoder={ # add ablation parameters to encoder
        'no local context': noLocalContext,
        'no global context': noGlobalContext,
        **intermediateContexts,
      }
    )
    return self._model(reverseArgs=reverseArgs, **kwargs)
  
  @property
  def kind(self): return self._model.kind
  
  @property
  def name(self): return self._model.name
# End of CDiffusionModelProxy
