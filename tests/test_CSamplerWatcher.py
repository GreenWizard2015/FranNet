import tensorflow as tf
from Utils.utils import CFakeObject
from NN.restorators.samplers import sampler_from_config
from NN.restorators.samplers.CSamplerWatcher import CSamplerWatcher

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

def _fake_AR(threshold, timesteps=10):
  interpolant = sampler_from_config({
    "name": "autoregressive",
    "noise provider": "normal",
    "threshold": threshold,
    "steps": {
      "start": 1.0,
      "end": 0.001,
      "steps": timesteps,
      "decay": 0.9
    },
    "interpolant": { "name": "direction" }
  })
 
  shape = (32, 3)
  fakeNoise = tf.random.normal(shape)
  def fakeModel(x, t, mask, **kwargs):
    s = tf.boolean_mask(fakeNoise, mask)
    return s + tf.cast(t, tf.float32) * x

  x = tf.random.normal(shape)
  return CFakeObject(x=x, model=fakeModel, interpolant=interpolant)

def test_notAffectResults():
  fake = _fake_sampler()
  X_withoutWatcher = fake.interpolant.sample(value=fake.x, model=fake.model)

  watcher = CSamplerWatcher(steps=10, tracked=dict())
  X_withWatcher = fake.interpolant.sample(value=fake.x, model=fake.model, algorithmInterceptor=watcher.interceptor())

  tf.debugging.assert_equal(X_withoutWatcher, X_withWatcher)
  return

def _commonChecks(collectedSteps, x, results):
  tf.debugging.assert_equal(tf.shape(collectedSteps)[1:], tf.shape(x))
  tf.debugging.assert_equal(collectedSteps[0], x, 'First step must be equal to initial value')
  tf.debugging.assert_equal(collectedSteps[-1], results, 'Last step must be equal to results')
  return

def test_collectsSteps():
  fake = _fake_sampler()
  watcher = CSamplerWatcher(
    steps=10,
    tracked=dict(
      value=(32, 3)
    )
  )
  oldX = watcher.tracked('value').numpy()
  results = fake.interpolant.sample(value=fake.x, model=fake.model, algorithmInterceptor=watcher.interceptor())

  collectedSteps = watcher.tracked('value')
  tf.debugging.assert_greater(
    tf.reduce_sum(tf.abs(collectedSteps - oldX)), 0.0,
    'Tracked steps must be different from initial value'
  )
  _commonChecks(collectedSteps, fake.x, results)
  tf.debugging.assert_equal(tf.shape(collectedSteps)[0], 11, 'Must collect 11 values')
  tf.debugging.assert_equal(watcher.iteration, 10, 'Must collect 10 steps')
  return

def test_collectsOnlyIndicedValues():
  fake = _fake_sampler()
  indices = [0, 2, 6]
  watcher = CSamplerWatcher(
    steps=10,
    tracked=dict(
      value=(3,)
    ),
    indices=indices
  )
  results = fake.interpolant.sample(value=fake.x, model=fake.model, algorithmInterceptor=watcher.interceptor())

  collectedSteps = watcher.tracked('value')
  _commonChecks(
    collectedSteps, 
    x=tf.gather(fake.x, indices, axis=0),
    results=tf.gather(results, indices, axis=0)
  )
  tf.debugging.assert_equal(tf.shape(collectedSteps)[0], 11, 'Must collect 11 values')
  return

def test_resetIteration():
  watcher = CSamplerWatcher(steps=10, tracked=dict())
  fake = _fake_sampler(timesteps=10)
  _ = fake.interpolant.sample(value=fake.x, model=fake.model, algorithmInterceptor=watcher.interceptor())
  tf.debugging.assert_equal(watcher.iteration, 10, 'Must collect 10 steps')
  
  fake = _fake_sampler(timesteps=5)
  _ = fake.interpolant.sample(value=fake.x, model=fake.model, algorithmInterceptor=watcher.interceptor())
  tf.debugging.assert_equal(watcher.iteration, 5, 'Must collect 5 steps')
  return

def _checkTracked(value, N):
  assert value is not None, 'Must be tracked'
  tf.debugging.assert_equal(tf.shape(value), (N, 32, 3), 'Unexpected shape')
  tf.debugging.assert_greater(tf.reduce_sum(tf.abs(value[0] - value[1])), 0.0, 'Must be different')
  return

def test_trackSolution():
  fake = _fake_sampler()
  watcher = CSamplerWatcher(
    steps=10,
    tracked=dict(x0=(32, 3), x1=(32, 3), value=(32, 3))
  )
  _ = fake.interpolant.sample(value=fake.x, model=fake.model, algorithmInterceptor=watcher.interceptor())
  _checkTracked(watcher.tracked('x0'), N=10)
  _checkTracked(watcher.tracked('x1'), N=10)
  _checkTracked(watcher.tracked('value'), N=11)
  return

def test_trackSolutionWithMask():
  fake = _fake_AR(threshold=0.1)
  watcher = CSamplerWatcher(
    steps=10,
    tracked=dict(x0=(32, 3), x1=(32, 3))
  )
  _ = fake.interpolant.sample(value=fake.x, model=fake.model, algorithmInterceptor=watcher.interceptor())
  _checkTracked(watcher.tracked('x0'), N=10)
  _checkTracked(watcher.tracked('x1'), N=10)
  return

def test_trackSolutionWithMask_value():
  fake = _fake_AR(threshold=0.1)
  watcher = CSamplerWatcher(
    steps=10,
    tracked=dict(value=(32, 3))
  )
  _ = fake.interpolant.sample(value=fake.x, model=fake.model, algorithmInterceptor=watcher.interceptor())
  _checkTracked(watcher.tracked('value'), N=11)
  return