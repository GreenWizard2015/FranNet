
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