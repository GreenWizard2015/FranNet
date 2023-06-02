import argparse, os, sys, json
# add the root folder of the project to the path
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../../'))

import numpy as np
from Utils.CDDIMParameters import DDIMParameters

def parametersFromArgs(args):
  # main combinations of parameters
  StochRange = np.linspace(0.0, 1.0, 11)[::-1]
  KRange = list(range(1, 11))
  parameters = []
  for stddev in ['zero', 'normal', 'squared']:
    for stochasticity in StochRange:
      for K in KRange:
        parameters.append(DDIMParameters(stochasticity=stochasticity, stddev=stddev, K=K))
        continue
      continue
    continue
  # some variations with clipping
  clipping = dict(min=-1.0, max=1.0)
  for stddev in ['zero', 'normal', 'squared']:
    for stochasticity in StochRange:
      for K in [1]:
        parameters.append(DDIMParameters(stochasticity=stochasticity, stddev=stddev, K=K, clipping=clipping))
        continue
      continue
    continue
  # with noise projection, no clipping, subset of stochasticity [0.0, 0.5, 1.0]
  for stddev in ['zero', 'normal', 'squared']:
    for stochasticity in [0.0, 0.5, 1.0]:
      for K in KRange:
        parameters.append(DDIMParameters(stochasticity=stochasticity, stddev=stddev, K=K, projectNoise=True))
        continue
      continue
    continue
  # with noise projection, subset of stochasticity [0.0, 0.5, 1.0], K = [15, 20, 25, 30], squared noise
  for stddev in ['squared']:
    for stochasticity in [0.0, 0.5, 1.0]:
      for K in [15, 20, 25, 30]:
        parameters.append(DDIMParameters(stochasticity=stochasticity, stddev=stddev, K=K, projectNoise=True))
        continue
      continue
    continue
  return parameters

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Generate configs for DDIM data collection')
  parser.add_argument('output', type=str, help='Path to output file')

  args = parser.parse_args()
  data = parametersFromArgs(args)
  print('Generated {} configs'.format(len(data)))
  with open(args.output, 'w') as f:
    json.dump(data, f, indent=2)
  pass