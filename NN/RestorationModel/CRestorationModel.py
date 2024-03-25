import tensorflow as tf

'''
wrapper which combines the encoders, decoder and restorator together 
to perform the restoration process
'''
class CRestorationModel(tf.keras.Model):
  def __init__(self,
    decoder, restorator,
    posEncoder, timeEncoder,
    **kwargs
  ):
    assert posEncoder is not None, "posEncoder is not provided"
    assert timeEncoder is not None, "timeEncoder is not provided"
    super().__init__(**kwargs)
    self._decoder = decoder
    self._restorator = restorator
    self._posEncoder = posEncoder
    self._timeEncoder = timeEncoder
    return
  
  def _encodeTime(self, t, training):
    t = tf.cast(t, tf.float32) # ensure that t is a tensor, not a nested structure
    tf.debugging.assert_less_equal(tf.reduce_max(t), 1.0)
    tf.debugging.assert_less_equal(0.0, tf.reduce_min(t))
    return self._timeEncoder(t, training=training)
  
  def _encodePos(self, pos, training, args):
    # for ablation study of the decoder, randomize positions BEFORE encoding
    if args.get('randomize positions', False):
      pos = tf.random.uniform(tf.shape(pos), minval=0.0, maxval=1.0)

    tf.debugging.assert_less_equal(tf.reduce_max(pos), 1.0)
    tf.debugging.assert_less_equal(0.0, tf.reduce_min(pos))
    return self._posEncoder(pos, training=training)
  
  def call(self, latents, pos, T, V, residual, training=None):
    EPos = self._encodePos(pos, training=training, args={})
    t = self._encodeTime(T, training=training)
    res = self._decoder(condition=latents, coords=EPos, timestep=t, V=V, training=training)
    return res + residual
  
  def reverse(self, latents, pos, reverseArgs, training, value, residual):
    EPos = self._encodePos(pos, training=training, args=reverseArgs.get('decoder', {}))

    def denoiser(x, t, mask=None):
      args = dict(condition=latents, coords=EPos, timestep=t, V=x)
      residuals = residual
      if mask is not None:
        args = {k: tf.boolean_mask(v, mask) for k, v in args.items()}
        residuals = tf.boolean_mask(residual, mask)

      return self._decoder(**args, training=training) + residuals
    
    return self._restorator.reverse(
      value=value, denoiser=denoiser, 
      modelT=lambda t: self._encodeTime(t, training=training),
      **reverseArgs
    )
  
  def train_step(self, x0, latents, positions, params, xT=None):
    residual = params['residual']
    params = {k: v for k, v in params.items() if k not in ['residual']}
    # defer training to the restorator
    return self._restorator.train_step(
      x0=x0,
      xT=xT,
      model=lambda T, V: self(
        latents=latents, pos=positions,
        T=T, V=V, residual=residual,
        training=True
      ),
      **params
    )
# End of CRestorationModel class