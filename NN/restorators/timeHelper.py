# Contains some functions for encoding time
import tensorflow as tf

def make_time_encoder(modelT):
  if modelT is None: # just in case
    modelT = lambda x: x
  
  encodedZeroT = modelT(tf.zeros((1, 1)))
  def _encodeTimeForModel(t, B):
    if t is None:
      return tf.repeat(encodedZeroT, repeats=B, axis=0)

    # if t is scalar
    if len(t.shape) == 0:
      encodedT = modelT( tf.reshape(t, (1, 1)) )
      return tf.repeat(encodedT, repeats=B, axis=0)
    
    # encode each unique time
    uniq = tf.unique( tf.reshape(t, (-1,)) )
    encodedT = modelT(uniq.y[:, None])
    
    M = tf.shape(encodedT)[1]
    T = tf.gather(encodedT, uniq.idx)
    tf.assert_equal(tf.shape(T), (B, M))
    return T
  
  return _encodeTimeForModel

def make_discrete_time_encoder(modelT, allT):
  if modelT is None: # just in case
    modelT = lambda x: x
  
  encodedT = modelT(allT)
  M = tf.shape(encodedT)[1]

  def _encodeTimeForModel(t, B):
    t = tf.squeeze(t)
    # if t is scalar
    if len(t.shape) == 0:
      T = encodedT[t, None]
      T = tf.repeat(T, repeats=B, axis=0)
      tf.assert_equal(tf.shape(T), (B, M))
      return T
    raise Exception('Should not be here')
  
  return _encodeTimeForModel