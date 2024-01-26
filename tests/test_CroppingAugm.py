import tensorflow as tf
import Utils.CroppingAugm as CroppingAugm

#############
# helper functions
def dummyImage(shape=(8, 34, 33, 3), dtype=tf.float32):
  return tf.random.uniform(shape, minval=0, maxval=1, dtype=dtype)

def assertSubsampled(res, crop_size, B, N):
  assert 'src' in res, 'src not in res'
  assert 'sampled' in res, 'sampled not in res'
  assert 'positions' in res, 'positions not in res'

  assert res['src'].shape == (B, crop_size, crop_size, 3)
  assert res['sampled'].shape == (B, N, 3)
  assert res['positions'].shape == (B, N, 2)
  return

def commonSampledCropTest(F):
  img = dummyImage()
  B, N = img.shape[0], 10
  crop_size = img.shape[1] // 2

  subsample = CroppingAugm.SubsampleProcessor(crop_size, N)
  res = F(img, crop_size, subsample)
  assertSubsampled(res, crop_size, B, N)
  return
#############
# center square crop returns the center of the image in dict
def test_centerSquareCrop_simple():
  img = dummyImage()
  crop_size = 16
  src_size = crop_size // 2
  res = CroppingAugm._centerSquareCrop(
    img, crop_size,
    CroppingAugm.RawProcessor(src_size)
  )
  assert res['src'].shape == (img.shape[0], src_size, src_size, img.shape[-1])
  assert res['dest'].shape == (img.shape[0], crop_size, crop_size, img.shape[-1])
  return

def test_centerSquareCrop_subsampled():
  return commonSampledCropTest(CroppingAugm._centerSquareCrop)

def test_randomSharedSquareCrop_subsampled():
  return commonSampledCropTest(CroppingAugm._randomSharedSquareCrop)

def test_randomSquareCrop_subsampled():
  return commonSampledCropTest(
    lambda img, crop_size, processor: CroppingAugm._randomSquareCrop(
      img, crop_size, 0.5, 1.0, processor
    )
  )

def test_ultraGridCrop_subsampled():
  return commonSampledCropTest(
    lambda img, crop_size, processor: CroppingAugm._ultraGridCrop(
      img, crop_size, 0.5, 1.0, processor
    )
  )

def test_ultraGridCrop():
  img = dummyImage()
  crop_size = img.shape[1] // 2
  src_size = crop_size // 2
  res = CroppingAugm._ultraGridCrop(
    img, crop_size, 0.5, 1.0,
    CroppingAugm.RawProcessor(src_size)
  )
  assert res['src'].shape == (img.shape[0], src_size, src_size, img.shape[-1])
  assert res['dest'].shape == (img.shape[0], crop_size, crop_size, img.shape[-1])
  return

def test_randomSquareCrop():
  img = dummyImage()
  crop_size = img.shape[1] // 2
  src_size = crop_size // 2
  res = CroppingAugm._randomSquareCrop(
    img, crop_size, 0.5, 1.0,
    CroppingAugm.RawProcessor(src_size)
  )
  assert res['src'].shape == (img.shape[0], src_size, src_size, img.shape[-1])
  assert res['dest'].shape == (img.shape[0], crop_size, crop_size, img.shape[-1])
  return

##########################################
def _verifyConfig(config):
  img = dummyImage()
  B, C = img.shape[0], img.shape[-1]
  config['crop size'] = crop_size = 23
  target_crop_size = 15
  # no subsampling
  cropper = CroppingAugm.configToCropper(
    dict(**config, subsample=False),
    dest_size=target_crop_size
  )
  res = cropper(img)
  assert res['src'].shape == (B, target_crop_size, target_crop_size, C)
  assert res['dest'].shape == (B, crop_size, crop_size, C)
  # with subsampling
  cropper = CroppingAugm.configToCropper(
    dict(**config, subsample={'N': 13}),
    dest_size=target_crop_size
  )
  res = cropper(img)
  assert 'dest' not in res, 'dest in res'
  assertSubsampled(res, target_crop_size, B, 13)
  return

# test configToCropper
def test_configToCropper_sharedCenterCrop():
  _verifyConfig({
    'random crop': False,
    'shared crops': True,
  })
  return

def test_configToCropper_sharedRandomCrop():
  _verifyConfig({
    'random crop': True,
    'shared crops': True,
  })
  return

def test_configToCropper_randomCrop():
  _verifyConfig({
    'random crop': True,
    'shared crops': False,
  })
  return

def test_configToCropper_ultraGridCrop():
  _verifyConfig({
    'random crop': True,
    'shared crops': False,
    'ultra grid': True,
  })
  return