from Utils.utils import setupGPU, load_config, setGPUMemoryLimit, merge_configs, JSONHelper
setupGPU() # call it on startup to prevent OOM errors on my machine

import os, argparse, shutil, re
from collections import defaultdict
from NN import model_from_config
from Utils.visualize import generateImage, data_from_dataset, data_from_input, makeImageProcessor
from Utils.WandBUtils import CWBProject

def _processData(data, model, processImage):
  NB_BATCHES = len(data)
  for batchId, batch in enumerate(data):
    print(f'Batch {batchId}/{NB_BATCHES}....')
    (srcB, dstB) = batch
    upscaledB = model(srcB)
    print(f'Upscaled shape: {upscaledB.shape}')
    # all have the same first dimension
    N = len(upscaledB)
    assert (N == len(srcB)) and (N == len(dstB)), f'srcB={len(srcB)}, dstB={len(dstB)}, upscaledB={len(upscaledB)}'

    for i in range(0, N):
      yield {
        'original': processImage(dstB[i]),
        'input': processImage(srcB[i]),
        'upscaled': processImage(upscaledB[i])[..., ::-1] # RGB -> BGR
      }
    continue
  return

def process(folder, visualizationConfig, model, modelArgsOverride, datasetProvider):
  if not os.path.exists(folder): os.makedirs(folder)
  data, dataset = datasetProvider(model.get_input_shape()[1:])
  ##############################
  modelArgs = {} # inference args: size, scale, shift, reverseArgs
  generationParams = {
    'mode': 'side by side',
    'format': 'png',
    'resize': 'resize',
  }
  
  if not(visualizationConfig is None):
    modelArgs.update(visualizationConfig['model args'])
    generationParams = visualizationConfig.get('output', generationParams)
  # force override model args
  modelArgs.update(modelArgsOverride)
  ##############################
  dataIttr = _processData(
    data,
    model=lambda x: model(x, **modelArgs),
    processImage=makeImageProcessor(dataset.unnormalizeImg)
  )
  for index, data in enumerate(dataIttr):
    generateImage(
      data=data,
      folder=folder,
      index=index,
      params=generationParams
    )
    continue
  return

def datasetFrom(args, config):
  if args.input is None:
    return lambda input_shape: data_from_dataset(config)
  # otherwise from input
  return lambda input_shape: data_from_input(args.input, input_shape)

def modelArgsOverrideFromArgs(args):
  modelArgs = {}
  if args.target_size:
    modelArgs['size'] = args.target_size

  if args.renderer_batch_size:
    modelArgs['batchSize'] = args.renderer_batch_size
  return modelArgs

def _bestRuns(runs):
  byName = defaultdict(list)
  for run in runs:
    byName[run.name].append(run)
    continue
  # in each group select the best run
  return [min(runs, key=lambda run: run.bestLoss) for runs in byName.values()]

def escape_directory_name(directory_name):
  directory_name = directory_name.strip()
  directory_name = re.sub(r'[\.<>:"/\\|?*]', '_', directory_name)
  return directory_name

def modelsFromArgs(args, config):
  if args.model: # load from file
    model = model_from_config(config['model'])
    model.load_weights(args.model)
    print('Model loaded successfully.')
    return [(model, None, config)]
  # otherwise from wandb project
  project = CWBProject(args.wandb_project)

  def isAccepted(runName):
    runName = runName.lower()
    return args.wandb_run_name.lower() in runName
  
  acceptedRuns = [run for run in project.runs() if isAccepted(run.name)]
  assert len(acceptedRuns) > 0, f'No runs found for {args.wandb_run_name}'
  if args.wandb_only_best: acceptedRuns = _bestRuns(acceptedRuns)
  print(f'Found {len(acceptedRuns)} runs for {args.wandb_run_name}:')
  for run in acceptedRuns: print(f'  {run.name} ({run.fullId}, loss: {run.bestLoss})')

  # ensure that user wants to continue and use from all selected runs evaluate the best model
  answer = input('Continue? [y/n]: ')
  assert answer.lower() == 'y', 'Aborted by user'

  byName = defaultdict(list)
  for run in acceptedRuns: byName[run.name].append(run)
  # return iterator of (model, modelArgs)
  configs = [config] if not isinstance(config, list) else config
  for runGroup in byName.values():
    for run in runGroup:
      print(f'Loading model from {run.name} ({run.fullId})...')
      bestModel = run.bestModel.pathTo()
      parts = [escape_directory_name(run.name)]
      if 1 < len(runGroup): parts.append(escape_directory_name(run.id))

      for i, config in enumerate(configs):
        modelConfigs = merge_configs(run.config, config) # override config with run config
        model = model_from_config(modelConfigs['model'])
        model.load_weights(bestModel)

        outputFolder = list(parts)
        if 1 < len(configs):
          folderName = modelConfigs.get('folder name', f'config_{i}')
          outputFolder.append(folderName)
        
        outputFolder = os.path.join(*outputFolder)
        yield (model, outputFolder, modelConfigs)
      continue
    continue
  return

def alterConfigFromArgs(args, config):
  if args.alter_config is None: return config
  # otherwise load from file set of configs and return array of configs
  alterConfig = JSONHelper.load(args.alter_config)
  assert isinstance(alterConfig, list), 'Alter config must be a list of configs'
  return [
    merge_configs(config, c) # override config with c
    for c in alterConfig
  ]

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Process arguments.')
  parser.add_argument(
    '--config', type=str, required=True,
    help='Path to a single config file or a multiple config files (they will be merged in order of appearance)',
    default=[], action='append', 
  )
  # file with a set of configs to be altered
  parser.add_argument('--alter-config', type=str, help='Path to a config file with a set of configs to be altered (optional)')
  parser.add_argument('--model', type=str, help='Path to model weights file')
  parser.add_argument('--target-size', type=int, help='Target size (optional)')
  parser.add_argument('--input', type=str, help='Path to image file or folder (optional)', default=None)
  # misc
  parser.add_argument('--folder', type=str, help='Path to output folder (optional)', default=None)
  parser.add_argument('--gpu-memory-mb', type=int, help='GPU memory limit in Mb (optional)')
  parser.add_argument('--renderer-batch-size', type=int, help='Renderer batch size (optional)')
  # wandb integration
  parser.add_argument('--wandb-project', type=str, help='Wandb project full name (entity/project) (optional)')
  parser.add_argument('--wandb-run-name', type=str, help='Wandb run name (optional, requires --wandb-project)')
  parser.add_argument('--wandb-only-best', action='store_true', help='Only use best model from wandb run (optional)')
  ########################### 
  args = parser.parse_args()
  # validate
  if args.wandb_project:
    assert args.wandb_run_name is not None, 'wandb-project requires wandb-run-name'
  assert (args.model is not None) or (args.wandb_project is not None), 'either model weights or wandb-project is required'
  assert (args.model is None) or (args.wandb_project is None), 'cannot use both model weights and wandb-project'
  ###########################
  if args.gpu_memory_mb: setGPUMemoryLimit(args.gpu_memory_mb)

  folder = os.getcwd()
  if args.folder:
    folder = os.path.abspath(args.folder)
    # clear/create folder
    if os.path.exists(folder): shutil.rmtree(folder)
    os.makedirs(folder)
    pass

  config = load_config(args.config, folder=os.getcwd())
  # should be specified input flag or config contains 'dataset' section
  assert (args.input is not None) or ('dataset' in config), 'either input or dataset section in config is required'
  
  models = modelsFromArgs(
    args=args,
    config=alterConfigFromArgs(args, config),    
  )
  for model, savePath, modelConfig in models:
    savePath = os.path.join(folder, savePath) if savePath else folder
    # clear/create folder
    if os.path.exists(savePath): shutil.rmtree(savePath)
    os.makedirs(savePath)
    
    print('Model loaded successfully. Output folder:', savePath)
    # save model config for future reference
    JSONHelper.save(os.path.join(savePath, 'model.json'), modelConfig)

    process(
      model=model,
      folder=savePath,
      modelArgsOverride=modelArgsOverrideFromArgs(args),
      datasetProvider=datasetFrom(args, config),
      visualizationConfig=config.get('visualization', None),
    )
    continue
  print('Done.')
  pass