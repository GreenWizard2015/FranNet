# this script performs following steps:
# 1. loads a diffusion model from wandb
# 2. evaluates it on the test set
# 3. converts the model to an interpolant
# 4. evaluates the interpolant on the test set
# 5. prints the results
import argparse, os, sys, json
# add the root folder of the project to the path
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../'))

from Utils.utils import setupGPU, load_config, merge_configs, JSONHelper
setupGPU() # call it on startup to prevent OOM errors on my machine

from Utils import dataset_from_config
from NN import model_from_config
from Utils.WandBUtils import CWBRun
from NN.restorators import replace_diffusion_restorator_by_interpolant

def modelProviderFromArgs(args, config):
  run = CWBRun(args.wandb_run, None, 'tmp')

  modelConfigs = merge_configs(run.config, config) # override config with run config
  restorator = modelConfigs['model']['restorator']
  assert restorator['name'] == 'diffusion', 'Must be a diffusion model'
  weights = run.models()[-2].pathTo()
  modelNet = model_from_config(modelConfigs['model']) # all models in the run should have the same config
  modelNet.load_weights(weights)
  yield(modelNet, run.name, run.fullId)

  # convert to interpolant
  modelConfigs['model']['restorator'] = replace_diffusion_restorator_by_interpolant(restorator)
  modelNet = model_from_config(modelConfigs['model'])
  modelNet.load_weights(weights)
  yield(modelNet, run.name + ' (interpolant)', run.fullId)
  return

def datasetProviderFromArgs(args, configs):
  def datasetProvider():
    dataset = dataset_from_config(configs['dataset'])
    return dataset.make_dataset(configs['dataset']['test'], split='test')
  return datasetProvider

def evaluateModel(model, datasetProvider):
  return model.evaluate(datasetProvider(), return_dict=True, verbose=1)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Evaluate best model from wandb run with different DDIM parameters')
  parser.add_argument(
    '--config', type=str, required=True,
    help='Path to a single config file or a multiple config files (they will be merged in order of appearance)',
    default=[], action='append', 
  )
  parser.add_argument('--wandb-run', type=str, help='Wandb run full id (entity/project/run_id)', required=True)
  
  args = parser.parse_args()
  configs = load_config(args.config, folder=os.getcwd())
  
  assert 'dataset' in configs, 'No dataset config found'
  datasetProvider = datasetProviderFromArgs(args, configs)

  results = []
  for model, modelName, runId in modelProviderFromArgs(args, configs):
    print(f'Starting evaluation of model "{modelName}" ({runId})')
    losses = evaluateModel(model, datasetProviderFromArgs(args, configs))
    results.append(dict(**losses, model=modelName, runId=runId))
    print()
    continue

  # print results
  results = sorted(results, key=lambda x: x['loss'])
  for r in results:
    print(f'{r["model"]} ({r["runId"]}) | loss: {r["loss"]}')
    continue
  print()
  pass