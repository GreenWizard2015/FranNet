import tensorflow as tf
from NN.restorators.diffusion.diffusion_samplers import sampler_from_config
from NN.restorators.diffusion.diffusion_schedulers import CDPDiscrete, get_beta_schedule

# test that DDPM and DDIM return same samples, when:
# - noise is zero/same
# - K=1, stochasticity=1.0
def _fake_samplers():
  schedule = CDPDiscrete( beta_schedule=get_beta_schedule('linear'), noise_steps=10 )
  ddim = sampler_from_config({
    'name': 'DDIM',
    'stochasticity': 1.0,
    'direction scale': 1.0,
    'noise stddev': 'zero',
    'steps skip type': { 'name': 'uniform', 'K': 1 },
  })
  ddpm = sampler_from_config({ 'name': 'DDPM', 'noise stddev': 'zero', })
  
  x = tf.random.normal([32, 3])
  fakeNoise = tf.random.normal([32, 3])
  def fakeModel(x, t):
    # check range of t
    tf.debugging.assert_less_equal(t, schedule.noise_steps - 1)
    tf.debugging.assert_greater_equal(t, 0)
    # any noticable perturbation will lead to different samples
    return fakeNoise + tf.cast(t + 1, tf.float32) * x
  return { 'ddim': ddim, 'ddpm': ddpm, 'schedule': schedule, 'x': x, 'fakeModel': fakeModel, 'fakeNoise': fakeNoise }

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

  for t in reversed(range(schedule.noise_steps)):
    t = tf.fill((32, 1), t)
    ddimS = ddimStepF(x=x, t=t, tPrev=t - 1)
    X_ddpm, var_ddpm = ddpmStepF(x=x, t=t)
    
    tf.debugging.assert_near(ddimS.x_prev, X_ddpm, atol=1e-6)
    tf.debugging.assert_near(ddimS.variance, var_ddpm, atol=1e-6)
    continue
  # last step should always have zero variance
  tf.assert_equal(ddimS.variance, 0.0)
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