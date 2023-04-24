from Utils.utils import setupGPU, load_config, setGPUMemoryLimit
setupGPU() # call it on startup to prevent OOM errors on my machine

import argparse, os, shutil, json
import tensorflow as tf
from NN import model_from_config, model_to_architecture
from Utils import dataset_from_config

def main(args):
  folder = os.path.dirname(__file__)
  config = load_config(args.config, folder=folder)
  assert "experiment" in config, "Config must contain 'experiment' key"
  
  if args.folder:
    folder = os.path.abspath(args.folder)
    # clear or create folder
    if os.path.exists(folder): shutil.rmtree(folder)
    os.makedirs(folder)

  # Override renderer batch size if specified
  if args.renderer_batch_size:
    config['model']['renderer']['batch_size'] = args.renderer_batch_size

  # Override train limit if specified
  if args.train_limit:
    config['dataset']['train']['limit'] = args.train_limit

  # Select dataset
  dataset = dataset_from_config(config['dataset'])
  train_data = dataset.make_dataset(config['dataset']['train'], split='train')
  test_data = dataset.make_dataset(config['dataset']['test'], split='test')

  # Create model
  model = model_from_config(config["model"], compile=True)
  model.summary(expand_nested=True)
  # save to config model architecture and number of parameters
  config['architecture'] = model_to_architecture(model)
  
  # Load weights if specified and evaluate
  if args.model:
    model.load_weights(args.model)
    model.evaluate(test_data)
    pass

  if args.dump_config:
    with open(args.dump_config, 'w') as f:
      json.dump(config, f, indent=2)
      pass
    pass

  if args.no_train: return

  latestModel = os.path.join(folder, 'model-latest.h5')
  callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
      filepath=os.path.join(folder, 'model-{epoch:02d}.h5'),
      save_weights_only=True, save_best_only=True, monitor='val_loss', verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
      filepath=latestModel,
      save_weights_only=True, save_best_only=True, monitor='val_loss', verbose=1
    ),
    tf.keras.callbacks.TerminateOnNaN(),
  ]

  if args.wandb: # init wandb
    import wandb
    wandb.init(project=args.wandb, entity=args.wandb_entity, config=config)
    callbacks.append(wandb.keras.WandbCallback(
      save_model=False, # save model to wandb manually
      save_graph=False,
    ))
    pass
  ########################
  try:
    # Fit model
    model.fit(
      train_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE),
      validation_data=test_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE),
      **config['fit_params'],
      callbacks=callbacks
    )
  finally:
    # if using wandb, save the best model to wandb
    if args.wandb:
      wandb.log_artifact(latestModel, type='bytes')
      wandb.finish()
  return

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Process arguments.')
  parser.add_argument(
    '--config', type=str, required=True,
    help='Path to a single config file or a multiple config files (they will be merged in order of appearance)',
    default=[], action='append', 
  )
  parser.add_argument('--model', type=str, help='Path to model weights file (optional)')
  parser.add_argument('--folder', type=str, help='Path to folder to save model (optional)')
  parser.add_argument('--train-limit', type=int, help='Limit number of training samples (optional)')
  parser.add_argument('--no-train', action='store_true', help='Do not train model (optional)')
  parser.add_argument('--gpu-memory-mb', type=int, help='GPU memory limit in Mb (optional)')
  parser.add_argument('--renderer-batch-size', type=int, help='Renderer batch size (optional)')
  parser.add_argument('--dump-config', type=str, help='Dump config to file (optional)')

  parser.add_argument('--wandb', type=str, help='Wandb project name (optional)')
  parser.add_argument('--wandb-entity', type=str, help='Wandb entity name (optional)')

  args = parser.parse_args()
  if args.gpu_memory_mb: setGPUMemoryLimit(args.gpu_memory_mb)
  main(args)
  pass