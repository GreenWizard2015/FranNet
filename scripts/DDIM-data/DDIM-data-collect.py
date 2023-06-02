import argparse, os, sys, json
# add the root folder of the project to the path
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../../'))

from Utils.utils import setupGPU, load_config, merge_configs, JSONHelper
setupGPU() # call it on startup to prevent OOM errors on my machine

from Utils import dataset_from_config
from NN import model_from_config
from Utils.WandBUtils import CWBRun

# TODO: Find a way to evaluate the model with different DDIM parameters without reloading it every time
def modelProviderFromArgs(args, configs):
  run = CWBRun(args.wandb_run, None, 'tmp')
  print(f'Loading model from {run.name} ({run.fullId})...')
  bestModel = run.bestModel.pathTo()
  print(f'Best model weights are stored at "{bestModel}"')

  def createModel(extraConfig):
    modelConfigs = merge_configs(run.config, extraConfig) # override config with run config
    model = model_from_config(modelConfigs['model'])
    model.load_weights(bestModel)
    return model
  return createModel

def datasetProviderFromArgs(args, configs):
  def datasetProvider():
    dataset = dataset_from_config(configs['dataset'])
    return dataset.make_dataset(configs['dataset']['test'], split='test')
  return datasetProvider

def collectData(modelProvider, datasetProvider, parameters):
  print(f'Found {len(parameters)} parameters to evaluate')
  def f():
    for i, (samplerConfig, params) in enumerate(parameters):
      print()
      print('%d / %d | Evaluate model with parameters: %s' % (i, len(parameters), ', '.join(
        [f'{k}={v}' for k, v in params.items()]
      )))
      # create model with new parameters
      model = modelProvider(samplerConfig)
      # evaluate model on test dataset
      losses = model.evaluate(datasetProvider(), return_dict=True, verbose=1)
      # save results
      yield dict(params, **losses)
      continue
    return

  return f

def filterData(parameters, data):
  oldData = []
  # filter out parameters that were already evaluated
  toKey = lambda x: json.dumps(x, sort_keys=True)
  byParams = { toKey(p[1]): p for p in parameters }
  for d in data:
    # d contains all parameters and losses, we only need the parameters
    key = toKey({ k: d[k] for k in d.keys() if not('loss' in k) })
    if key in byParams:
      oldData.append(d)
      del byParams[key]
    continue

  parameters = list(byParams.values())
  print(f'Found {len(parameters)} parameters that were not evaluated')
  print(f'Found {len(oldData)} parameters that were already evaluated')
  return parameters, oldData

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Evaluate best model from wandb run with different DDIM parameters')
  parser.add_argument(
    '--config', type=str, required=True,
    help='Path to a single config file or a multiple config files (they will be merged in order of appearance)',
    default=[], action='append', 
  )
  parser.add_argument('--wandb-run', type=str, help='Wandb run full id (entity/project/run_id)', required=True)
  parser.add_argument('--parameters', type=str, help='Path to parameters file', required=True)
  parser.add_argument('--output', type=str, help='Path to output folder (optional)', default='DDIM-data.json')
  
  args = parser.parse_args()
  parameters = JSONHelper.load(args.parameters)
  oldData = []
  if os.path.exists(args.output):
    parameters, oldData = filterData(parameters, JSONHelper.load(args.output))
    
  configs = load_config(args.config, folder=os.getcwd())
  data = collectData(
    modelProvider=modelProviderFromArgs(args, configs),
    datasetProvider=datasetProviderFromArgs(args, configs),
    parameters=parameters,
  )
  for newData in data():
    oldData.append(newData)
    JSONHelper.save(args.output, oldData)
    print(f'Saved {len(oldData)} data points to "{args.output}"')
    continue
  pass