import tensorflow as tf
import pytest
from NN.restorators.diffusion.diffusion_samplers import sampler_from_config
from NN.restorators.diffusion.diffusion_schedulers import CDPDiscrete, get_beta_schedule

def _fake_model(noise_steps):
  x = tf.random.normal([32, 3])
  fakeNoise = tf.random.normal([32, 3])
  def fakeModel(x, t):
    # check range of t
    tf.debugging.assert_less_equal(t, noise_steps - 1)
    tf.debugging.assert_greater_equal(t, 0)
    # any noticable perturbation will lead to different samples
    return fakeNoise + tf.cast(t + 1, tf.float32) * x
  return { 'x': x, 'fakeModel': fakeModel, 'fakeNoise': fakeNoise }

def _fake_DDIM(stochasticity, K, useFloat64=False, noiseProjection=False):
  return sampler_from_config({
    'name': 'DDIM',
    'stochasticity': stochasticity,
    'noise stddev': 'zero',
    'steps skip type': { 'name': 'uniform', 'K': K },
    'use float64': useFloat64,
    'project noise': noiseProjection,
  })

# test that DDPM and DDIM return same samples, when:
# - noise is zero/same
# - K=1, stochasticity=1.0
def _fake_samplers():
  schedule = CDPDiscrete( beta_schedule=get_beta_schedule('linear'), noise_steps=10 )
  ddim = _fake_DDIM(stochasticity=1.0, K=1)
  ddpm = sampler_from_config({ 'name': 'DDPM', 'noise stddev': 'zero', })
  model = _fake_model(schedule.noise_steps)
  return { 'ddim': ddim, 'ddpm': ddpm, 'schedule': schedule, **model }

def test_DDPM_eq_DDIM_steps():
  samplers = _fake_samplers()
  ddim = samplers['ddim']
  ddpm = samplers['ddpm']
  schedule = samplers['schedule']
  x = samplers['x']
  fakeModel = samplers['fakeModel']
  ########
  ddimStepF = ddim._reverseStep(fakeModel, schedule=schedule, eta=1.0)
  ddpmStepF = ddpm._reverseStep(fakeModel, schedule=schedule)

  for T in reversed(range(schedule.noise_steps)):
    t = tf.fill((32, 1), T)
    ddimS = ddimStepF(x=x, t=t, tPrev=t - 1)
    X_ddpm, var_ddpm = ddpmStepF(x=x, t=t)
    
    s = schedule.parametersForT(T)
    tf.print(T, s.alphaHat, s.sigma)
    tf.debugging.assert_near(ddimS.x_prev, X_ddpm, atol=1e-5, message=f"t={T}")
    tf.debugging.assert_near(ddimS.sigma, var_ddpm, atol=1e-6, message=f"t={T}")
    if 0 < T:
      tf.debugging.assert_greater(ddimS.sigma, 0.0, message=f"t={T}")
      tf.debugging.assert_greater(var_ddpm, 0.0, message=f"t={T}")
    continue
  # last step should always have zero variance
  tf.assert_equal(ddimS.sigma, 0.0)
  tf.assert_equal(var_ddpm, 0.0)
  return

def test_DDPM_eq_DDIM_sample():
  samplers = _fake_samplers()
  ddim = samplers['ddim']
  ddpm = samplers['ddpm']
  schedule = samplers['schedule']
  x = samplers['x']
  fakeModel = samplers['fakeModel']
  ########
  X_ddim = ddim.sample(value=x, model=fakeModel, schedule=schedule)
  X_ddpm = ddpm.sample(value=x, model=fakeModel, schedule=schedule)
  tf.debugging.assert_near(X_ddim, X_ddpm, atol=1e-6)
  return

def test_DDPM_eq_DDIM_sample_modelCalls():
  samplers = _fake_samplers()
  ddim = samplers['ddim']
  ddpm = samplers['ddpm']
  schedule = samplers['schedule']
  x = samplers['x']
  fakeModel = samplers['fakeModel']
  def makeCounter():
    def counter(*args, **kwargs):
      counter.calls.assign_add(1)
      return fakeModel(*args, **kwargs)
    counter.calls = tf.Variable(0, dtype=tf.int32)
    return counter
  
  fakeModelA = makeCounter()
  fakeModelB = makeCounter()
  ########
  _ = ddpm.sample(value=x, model=fakeModelA, schedule=schedule)
  _ = ddim.sample(value=x, model=fakeModelB, schedule=schedule)
  
  tf.assert_equal(fakeModelB.calls, fakeModelA.calls)
  tf.assert_equal(fakeModelA.calls, schedule.noise_steps)
  tf.assert_equal(fakeModelB.calls, schedule.noise_steps)
  return

def test_DDIM_float64():
  schedule = CDPDiscrete( beta_schedule=get_beta_schedule('linear'), noise_steps=10 )
  model = _fake_model(schedule.noise_steps)
  x, fakeModel = model['x'], model['fakeModel']

  ddimA = _fake_DDIM(stochasticity=1.0, K=1, useFloat64=False)
  ddimB = _fake_DDIM(stochasticity=1.0, K=1, useFloat64=True)

  A = ddimA.sample(value=x, model=fakeModel, schedule=schedule)
  B = ddimB.sample(value=x, model=fakeModel, schedule=schedule)

  tf.debugging.assert_near(A, B, atol=1e-6)
  return

# test that noise projection does not change if noise is zero
@pytest.mark.parametrize(
  'stochasticity,K',
  [
    (1.0, 1),
    (0.0, 2),
    (0.5, 3),
  ]
)
def test_DDIM_noiseProjection(stochasticity, K):
  schedule = CDPDiscrete( beta_schedule=get_beta_schedule('linear'), noise_steps=10 )
  model = _fake_model(schedule.noise_steps)
  x, fakeModel = model['x'], model['fakeModel']

  ddimA = _fake_DDIM(stochasticity=stochasticity, K=K, noiseProjection=False)
  ddimB = _fake_DDIM(stochasticity=stochasticity, K=K, noiseProjection=True)

  A = ddimA.sample(value=x, model=fakeModel, schedule=schedule)
  B = ddimB.sample(value=x, model=fakeModel, schedule=schedule)

  tf.debugging.assert_near(A, B, atol=1e-6)
  return

# verify that stochasticity has an effect even if noise is zero
def test_DDIM_stochasticity_effect():
  schedule = CDPDiscrete( beta_schedule=get_beta_schedule('linear'), noise_steps=10 )
  model = _fake_model(schedule.noise_steps)
  x, fakeModel = model['x'], model['fakeModel']

  ddimA = _fake_DDIM(stochasticity=0.0, K=1)
  ddimB = _fake_DDIM(stochasticity=1.0, K=1)

  A = ddimA.sample(value=x, model=fakeModel, schedule=schedule)
  B = ddimB.sample(value=x, model=fakeModel, schedule=schedule)

  tf.debugging.assert_greater(tf.reduce_mean(tf.abs(A - B)), 1e-3)
  return