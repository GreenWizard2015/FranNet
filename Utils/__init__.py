
def dataset_from_config(config):
  # Select dataset
  if config['name'] == 'celeba':
    from Utils.celeba import CCelebADataset
    return CCelebADataset(
      image_size=config['image_size'],
      batch_size=config['batch_size'],
      toGrayscale=config.get('toGrayscale', True),
    )

  raise NotImplementedError(f"Dataset {config['name']} not implemented.")

# Hacky way to create ImageProcessor without instantiating a dataset
def ImageProcessor_from_config(config):
  if isinstance(config, str) and ('celeba' == config.lower()): # shortcut
    config = dict(name='celeba', image_size=64, toGrayscale=True)

  if isinstance(config, dict) and ('celeba' == config['name'].lower()):
    from Utils.CelebaImageProcessor import CelebaImageProcessor
    return CelebaImageProcessor(
      image_size=config['image_size'],
      to_grayscale=config.get('toGrayscale', True),
    )
  
  raise NotImplementedError(f"ImageProcessor {config['name']} not implemented.")