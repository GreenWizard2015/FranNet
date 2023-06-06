import numpy as np
import json
from functools import wraps, lru_cache
from PIL import Image

# https://stackoverflow.com/questions/6358481/using-functools-lru-cache-with-dictionary-arguments
def hashable_lru(maxsize):
  def cacheMe(func):
    def func_with_serialized_params(*args, **kwargs):
      def deserialise(value):
        try:
          return json.loads(value)
        except Exception:
          return value
        return
      
      _args = tuple([deserialise(arg) for arg in args])
      _kwargs = {k: deserialise(v) for k, v in kwargs.items()}
      return func(*_args, **_kwargs)

    cache = lru_cache(maxsize=maxsize)
    cached_function = cache(func_with_serialized_params)

    @wraps(func)
    def lru_decorator(*args, **kwargs):
      _args = tuple([json.dumps(arg, sort_keys=True) if type(arg) in (list, dict) else arg for arg in args])
      _kwargs = {k: json.dumps(v, sort_keys=True) if type(v) in (list, dict) else v for k, v in kwargs.items()}
      return cached_function(*_args, **_kwargs)
    lru_decorator.cache_info = cached_function.cache_info
    lru_decorator.cache_clear = cached_function.cache_clear
    return lru_decorator
  
  return cacheMe

def toPILImage(img, isBGR=False):
  if not (np.uint8 == img.dtype): img = (img * 255).astype(np.uint8)
  if 2 == len(img.shape): img = img[..., None]
  if 1 == img.shape[-1]: img = np.repeat(img, 3, axis=-1) # grayscale
  if isBGR: img = img[..., ::-1]
  
  img = np.clip(img, 0, 255)

  assert 3 == len(img.shape), f'Invalid image shape: {img.shape}'
  assert np.uint8 == img.dtype, f'Invalid image dtype: {img.dtype}'
  assert 3 == img.shape[-1], f'Invalid image channels: {img.shape[-1]}'
  return Image.fromarray(img, 'RGB')
