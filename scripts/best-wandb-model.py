# this script scans the wandb run for the best with the given parameters and prints the results:
# wandb entity/project/run_id : loss
import argparse, os, sys, json
# add the root folder of the project to the path
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../'))

from Utils.utils import setupGPU, load_config, merge_configs, JSONHelper
setupGPU() # call it on startup to prevent OOM errors on my machine

from Utils import dataset_from_config
from NN import model_from_config
from Utils.WandBUtils import CWBRun

def modelProviderFromArgs(args, config):
  run = CWBRun(args.wandb_run, None, 'tmp')
  models = run.models()

  modelConfigs = merge_configs(run.config, config) # override config with run config
  modelNet = model_from_config(modelConfigs['model']) # all models in the run should have the same config
  for model in models:
    modelNet.load_weights( model.pathTo() )
    yield(modelNet, model.name, run.fullId)
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