import tensorflow as tf

class CSequentialRestorator(tf.keras.Model):
  def __init__(self, restorators, **kwargs):
    super().__init__(**kwargs)
    self._restorators = restorators
    return
  
  # only for training and building
  def call(self, latents, pos, T, V, residual, training=None):
    for restorator in self._restorators:
      V = restorator(
        latents=latents, pos=pos, T=T, V=V, residual=residual, training=training
      )
      continue
    return V
  
  def reverse(self, latents, pos, reverseArgs, training, value, residual):
    for restorator in self._restorators:
      if tf.is_tensor(value): value = tf.stop_gradient(value)
      value = restorator.reverse(
        latents=latents, pos=pos, reverseArgs=reverseArgs,
        residual=residual,
        training=training, value=value
      )
      continue
    return value
  
  def train_step(self, x0, latents, positions, params, xT=None):
    loss = 0.0
    # first restorator is trained as usual
    trainStep = self._restorators[0].train_step(
      x0=x0, xT=xT, latents=latents, positions=positions, params=params
    )
    loss += trainStep['loss']
    xT = tf.stop_gradient(trainStep['value'])
    # the rest of the restorators are trained sequentially
    for restorator in self._restorators:
      trainStep = restorator.train_step(
        x0=tf.stop_gradient(x0),
        xT=tf.stop_gradient(xT),
        latents=tf.stop_gradient(latents),
        positions=tf.stop_gradient(positions),
        params={k: tf.stop_gradient(v) if isinstance(v, tf.Tensor) else v for k, v in params.items()}
      )
      loss += trainStep['loss']
      xT = tf.stop_gradient(trainStep['value'])
      continue
    return dict(loss=loss, value=xT)
# End of CSequentialRestorator