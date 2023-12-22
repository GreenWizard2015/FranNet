import pytest
import tensorflow as tf
from Utils.utils import CFakeObject
from NN.restorators.samplers import sampler_from_config
from NN.restorators.samplers.CSamplerWatcher import CSamplerWatcher
from NN.restorators.samplers.CWatcherWithExtras import CWatcherWithExtras
from Utils import colors

def _fake_sampler(stochasticity=1.0, timesteps=10):
  interpolant = sampler_from_config({
    'name': 'DDIM',
    'stochasticity': stochasticity,
    'noise stddev': 'zero',
    'schedule': {
      'name': 'discrete',
      'beta schedule': 'linear',
      'timesteps': timesteps,
    },
    'steps skip type': { 'name': 'uniform', 'K': 1 },
  })
  
  shape = (32, 3)
  fakeNoise = tf.random.normal(shape)
  def fakeModel(x, T, **kwargs):
    return fakeNoise + tf.cast(T, tf.float32) * x

  x = tf.random.normal(shape)
  return CFakeObject(x=x, model=fakeModel, interpolant=interpolant)

def test_sameResults():
  fake = _fake_sampler()
  watcher = CSamplerWatcher(
    steps=10,
    tracked=dict(value=(32, 3), x0=(32, 3), x1=(32, 3))
  )
  resultsA = fake.interpolant.sample(value=fake.x, model=fake.model, algorithmInterceptor=watcher.interceptor())
  
  residuals = tf.random.normal((32, 3))
  watcherB = CWatcherWithExtras(watcher=watcher, converter=None, residuals=residuals)
  resultsB = fake.interpolant.sample(value=fake.x, model=fake.model, algorithmInterceptor=watcherB.interceptor())

  tf.debugging.assert_near(resultsA, resultsB)
  return

@pytest.mark.parametrize('field', ['value', 'x0', 'x1'])
def test_shiftedValues(field):
  fake = _fake_sampler()
  watcher = CSamplerWatcher(steps=10, tracked={field: (32, 3)})
  
  fake.interpolant.sample(value=fake.x, model=fake.model, algorithmInterceptor=watcher.interceptor())
  valuesA = watcher.tracked(field).numpy()
  
  residuals = tf.random.normal((32, 3))
  watcherB = CWatcherWithExtras(watcher=watcher, converter=None, residuals=residuals)
  fake.interpolant.sample(value=fake.x, model=fake.model, algorithmInterceptor=watcherB.interceptor())
  valuesB = watcherB.tracked(field).numpy()

  tf.debugging.assert_near(valuesA + residuals[None], valuesB)
  return

@pytest.mark.parametrize('field', ['value', 'x0', 'x1'])
def test_transformedValues(field):
  fake = _fake_sampler()
  watcher = CSamplerWatcher(steps=10, tracked={field: (32, 3)})
  converter = colors.convertRGBtoLAB()
  
  fake.interpolant.sample(value=fake.x, model=fake.model, algorithmInterceptor=watcher.interceptor())
  valuesA = watcher.tracked(field).numpy()
  
  residuals = tf.random.normal((32, 3))
  watcherB = CWatcherWithExtras(watcher=watcher, converter=converter, residuals=residuals)
  fake.interpolant.sample(value=fake.x, model=fake.model, algorithmInterceptor=watcherB.interceptor())
  valuesB = watcherB.tracked(field).numpy()

  tf.debugging.assert_near(converter.convertBack(valuesA) + residuals[None], valuesB)
  return