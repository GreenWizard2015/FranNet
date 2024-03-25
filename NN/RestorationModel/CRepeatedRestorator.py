import tensorflow as tf

class CRepeatedRestorator(tf.keras.Model):
  def __init__(self, restorator, IDs, N, **kwargs):
    super().__init__(**kwargs)
    self._restorator = restorator
    self._IDs = IDs
    self._N = N
    return
  
  def _withID(self, latents, idx, training):
    id = self._IDs(idx, training=training)
    B = tf.shape(latents)[0]
    id = tf.repeat(id, repeats=B, axis=0)
    res = tf.concat([latents, id], axis=-1)
    tf.assert_equal(tf.shape(res)[0], B)
    return res
  
  # only for training and building
  def call(self, latents, pos, T, V, residual, training=None):
    for i in range(self._N):
      V = self._restorator(
        latents=self._withID(latents, i, training),
        pos=pos, T=T, V=V, residual=residual, training=training
      )
      continue
    return V
  
  def reverse(self, latents, pos, reverseArgs, training, value, residual):
    for i in range(self._N):
      if tf.is_tensor(value): value = tf.stop_gradient(value)
      value = self._restorator.reverse(
        latents=self._withID(latents, i, training),
        pos=pos, reverseArgs=reverseArgs,
        residual=residual,
        training=training, value=value
      )
      continue
    return value
  
  def train_step(self, x0, latents, positions, params, xT=None):
    loss = 0.0
    trainStep = self._restorator.train_step(
      x0=x0, xT=xT,
      latents=self._withID(latents, 0, training=True),
      positions=positions, params=params
    )
    loss += trainStep['loss']

    for i in range(1, self._N):
      trainStep = self._restorator.train_step(
        x0=tf.stop_gradient(x0),
        xT=tf.stop_gradient(xT),
        latents=self._withID(tf.stop_gradient(latents), i, training=True),
        positions=tf.stop_gradient(positions),
        params={k: tf.stop_gradient(v) if tf.is_tensor(v) else v for k, v in params.items()}
      )
      loss += trainStep['loss']
      # add residual to obtain the new x0
      xT = tf.stop_gradient(trainStep['value'])
      continue
    return dict(loss=loss, value=xT)
# End of CRepeatedRestorator