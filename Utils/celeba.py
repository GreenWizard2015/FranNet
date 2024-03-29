import tensorflow_datasets as tfds
from Utils.utils import masking_from_config
from Utils.CelebaImageProcessor import CelebaImageProcessor

class CCelebADataset:
  def __init__(self, batch_size=32, image_size=64, toGrayscale=True):
    self._imageProcessor = CelebaImageProcessor(image_size, toGrayscale)
    self._batchSize = batch_size

    self._celeb_a_builder = tfds.builder("celeb_a")
    self._celeb_a_builder.download_and_prepare()
    return
  
  @property
  def range(self):
    return self._imageProcessor.range
  
  def make_dataset(self, config, split):
    res = self._celeb_a_builder.as_dataset(split=split)

    batch_size = config.get('batch_size', self._batchSize)
    limit = config.get('limit', None)
    if limit: res = res.take(limit)
    if batch_size: res = res.batch(batch_size)

    processF = self._imageProcessor.process(config)
    res = res.map(lambda x: processF(x['image']))

    if 'masking' in config: res = res.map( masking_from_config(config['masking']) )
    if 'repeat' in config: res = res.repeat(config['repeat'])
    if 'shuffle' in config: res = res.shuffle(config['shuffle'])
    return res
  
if __name__ == "__main__": # test masking
  import cv2
  dataset = CCelebADataset(image_size=64, batch_size=1)
  train = dataset.make_dataset(
    {
      'batch_size': 1, 'limit': 16,
      'masking': {
        'name': 'grid',
        'size': [5, 7, 11, 13, 17, 19, 23, 29, 31],
        'min': 0, 'max': 0.75, # min and max number of masked squares
        'mask value': -1.0, # normalized value of masked squares
      }
    }, 'train'
  )
  for src, img in train.take(12):
    src = src[0].numpy()
    img = img[0].numpy()
    src = dataset.range.convertBack(src)
    img = dataset.range.convertBack(img)
    # upscale src by 4x
    src = cv2.resize(src, (256, 256), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('src', src.astype('uint8'))
    cv2.imshow('img', img.astype('uint8')[..., ::-1])
    cv2.waitKey(0)
    pass
  pass