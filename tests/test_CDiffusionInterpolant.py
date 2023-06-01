import tensorflow as tf
import numpy as np
import pytest
from Utils.utils import CFakeObject
# 
from NN.restorators.diffusion.diffusion_samplers import sampler_from_config as diffusion_sampler_from_config
from NN.restorators.diffusion.diffusion_schedulers import schedule_from_config
from NN.restorators.samplers import sampler_from_config
from NN.restorators.interpolants.CDiffusionInterpolant import CDiffusionInterpolant, CDiffusionInterpolantV

def _fake_samplers(stochasticity, stepsConfig):
  scheduleConfig = {
    'name': 'discrete',
    'beta schedule': 'linear',
    'timesteps': 10
  }
  schedule = schedule_from_config(scheduleConfig)
  ddim = diffusion_sampler_from_config({
    'name': 'DDIM',
    'stochasticity': stochasticity,
    'direction scale': 1.0,
    'noise stddev': 'zero',
    'steps skip type': stepsConfig,
  })
  
  x = tf.random.normal([32, 3])
  fakeNoise = tf.random.normal([32, 3])
  def fakeModel(x, T, **kwargs):
    return fakeNoise + tf.cast(T, tf.float32) * x

  interpolant = sampler_from_config({
    'name': 'DDIM',
    'stochasticity': stochasticity,
    'noise stddev': 'zero',
    'schedule': scheduleConfig,
    'steps skip type': stepsConfig,
  })
  return CFakeObject(
    ddim=ddim,
    schedule=schedule,
    x=x,
    model=fakeModel,
    interpolant=interpolant
  )

@pytest.mark.parametrize('stochasticity', [0.0, 0.1, 0.5, 1.0])
def test_DDIM_eq_INTR_sample(stochasticity):
  fake = _fake_samplers(
    stochasticity,
    stepsConfig={ 'name': 'uniform', 'K': 1 }
  )
  
  X_ddim = fake.ddim.sample(value=fake.x, model=fake.model, schedule=fake.schedule)
  X_interpolant = fake.interpolant.sample(value=fake.x, model=fake.model)
  tf.debugging.assert_near(X_ddim, X_interpolant, atol=5e-6)
  return

@pytest.mark.parametrize('K', [2, 3, 5, 7])
def test_DDIM_eq_INTR_sample_steps(K):
  fake = _fake_samplers(
    stochasticity=0.1,
    stepsConfig={ 'name': 'uniform', 'K': K }
  )
  
  X_ddim = fake.ddim.sample(value=fake.x, model=fake.model, schedule=fake.schedule)
  X_interpolant = fake.interpolant.sample(value=fake.x, model=fake.model)
  tf.debugging.assert_near(X_ddim, X_interpolant, atol=5e-6)
  return

# Randomly sampled data go through train, and then we check if the solver is able to recover the data
def _check_inversibility(interpolant, N=1024, atol=5e-6):
  x0 = tf.zeros([N, 1])
  x1 = tf.ones([N, 1])
  T = tf.linspace(0.0, 1.0, N)
  T = tf.clip_by_value(T, 5e-6, 1.0 - 5e-6)
  trainData = interpolant.train(x0, x1, T[:, None])
  solved = interpolant.solve(x_hat=trainData['target'], xt=trainData['xT'], t=trainData['T'])

  T = T.numpy().reshape(-1)
  diff = tf.abs(solved.x0 - x0).numpy().reshape(-1)
  for i, (t, d) in enumerate(zip(T, diff)):
    assert np.allclose(0.0, d, atol=atol), f'{i}: {t} {d}'

  diff = tf.abs(solved.x1 - x1).numpy().reshape(-1)
  for i, (t, d) in enumerate(zip(T, diff)):
    assert np.allclose(0.0, d, atol=atol), f'{i}: {t} {d}'
  return

def test_inversibility():
  _check_inversibility(CDiffusionInterpolant(), N=1024 * 16, atol=5e-6)
  return

def test_inversibility_V():
  _check_inversibility(CDiffusionInterpolantV(), N=1024 * 16, atol=1e-5)
  return