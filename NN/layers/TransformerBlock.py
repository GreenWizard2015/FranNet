import tensorflow as tf
from absl import logging
ops=tf

def _check_masks_shapes(inputs, padding_mask, attention_mask):
    mask = padding_mask
    if hasattr(inputs, "_keras_mask") and mask is None:
        mask = inputs._keras_mask
    if mask is not None:
        if len(mask.shape) != 2:
            raise ValueError(
                "`padding_mask` should have shape "
                "(batch_size, target_length). "
                f"Received shape `{mask.shape}`."
            )
    if attention_mask is not None:
        if len(attention_mask.shape) != 3:
            raise ValueError(
                "`attention_mask` should have shape "
                "(batch_size, target_length, source_length). "
                f"Received shape `{mask.shape}`."
            )


def compute_causal_mask(batch_size, input_length, output_length, cache_index=0):
    """Compute a causal attention mask for a transformer decoder.

    Args:
        batch_size: batch size for the mask.
        input_length: the length of key/value tensors in the attention layer.
        output_length: the length of query tensors in the attention layer.
        cache_index: the current index for cached generation. If passed, the
            query sequence will be considered to start at `cache_index` rather
            than zero. For example, a causal mask with `output_length=1` and
            `cache_index=5` would allow the query tensor to attend to the first
            five positions of the key/value tensors.

    Return:
        A causal attention mask with shape
        `(batch_size, output_length, input_length)` that can be passed to a
        attention layer.
    """
    i = ops.arange(output_length, dtype="float32")
    i = i + ops.cast(cache_index, "float32")
    i = ops.expand_dims(i, axis=1)
    j = ops.arange(input_length, dtype="float32")
    mask = ops.expand_dims(i >= j, axis=0)

    return ops.broadcast_to(mask, (batch_size, output_length, input_length))


def merge_padding_and_attention_mask(
    inputs,
    padding_mask,
    attention_mask,
):
    """Merge the padding mask with a customized attention mask.

    Args:
        inputs: the input sequence.
        padding_mask: the 1D padding mask, of shape
            [batch_size, sequence_length].
        attention_mask: the 2D customized mask, of shape
            [batch_size, sequence1_length, sequence2_length].

    Return:
        A merged 2D mask or None. If only `padding_mask` is provided, the
        returned mask is padding_mask with one additional axis.
    """
    _check_masks_shapes(inputs, padding_mask, attention_mask)
    mask = padding_mask
    if hasattr(inputs, "_keras_mask"):
        if mask is None:
            # If no padding mask is explicitly provided, we look for padding
            # mask from the input data.
            mask = inputs._keras_mask
        else:
            logging.warning(
                "You are explicitly setting `padding_mask` while the `inputs` "
                "have built-in mask, so the built-in mask is ignored."
            )
    if mask is not None:
        # Add an axis for broadcasting, the attention mask should be 2D
        # (not including the batch axis).
        mask = ops.cast(ops.expand_dims(mask, axis=1), "int32")
    if attention_mask is not None:
        attention_mask = ops.cast(attention_mask, "int32")
        if mask is None:
            return attention_mask
        else:
            return ops.minimum(mask, attention_mask)
    return mask

def clone_initializer(initializer):
    """Clones an initializer to ensure a new seed.

    As of tensorflow 2.10, we need to clone user passed initializers when
    invoking them twice to avoid creating the same randomized initialization.
    """
    # If we get a string or dict, just return as we cannot and should not clone.
    if not isinstance(initializer, tf.keras.initializers.Initializer):
      return initializer
    config = initializer.get_config()
    return initializer.__class__.from_config(config)

class TransformerBlock(tf.keras.layers.Layer):
    """Transformer encoder.

    This class follows the architecture of the transformer encoder layer in the
    paper [Attention is All You Need](https://arxiv.org/abs/1706.03762). Users
    can instantiate multiple instances of this class to stack up an encoder.

    This layer will correctly compute an attention mask from an implicit
    Keras padding mask (for example, by passing `mask_zero=True` to a
    `tf.keras.layers.Embedding` layer). See the Masking and Padding
    [guide](https://keras.io/guides/understanding_masking_and_padding/)
    for more details.

    Args:
        intermediate_dim: int, the hidden size of feedforward network.
        num_heads: int, the number of heads in the
            `tf.keras.layers.MultiHeadAttention` layer.
        dropout: float. the dropout value, shared by
            `tf.keras.layers.MultiHeadAttention` and feedforward network.
            Defaults to `0.`.
        activation: string or `keras.activations`. the
            activation function of feedforward network.
            Defaults to `"relu"`.
        layer_norm_epsilon: float. The epsilon value in layer
            normalization components. Defaults to `1e-5`.
        kernel_initializer: string or `keras.initializers` initializer.
            The kernel initializer for the dense and multiheaded
            attention layers. Defaults to `"glorot_uniform"`.
        bias_initializer: string or `keras.initializers` initializer.
            The bias initializer for the dense and multiheaded
            attention layers. Defaults to `"zeros"`.
        normalize_first: bool. If True, the inputs to the
            attention layer and the intermediate dense layer  are normalized
            (similar to GPT-2). If set to False, outputs of attention layer and
            intermediate dense layer are normalized (similar to BERT).
            Defaults to `False`.
        name: string. The name of the layer. Defaults to `None`.
        **kwargs: other keyword arguments.

    Examples:

    ```python
    # Create a single transformer encoder layer.
    encoder = keras_nlp.layers.TransformerEncoder(
        intermediate_dim=64, num_heads=8)

    # Create a simple model containing the encoder.
    input =tf.keras.Input(shape=(10, 64))
    output = encoder(input)
    model =tf.keras.Model(inputs=input, outputs=output)

    # Call encoder on the inputs.
    input_data = np.random.uniform(size=(2, 10, 64))
    output = model(input_data)
    ```

    References:
     - [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
    """

    def __init__(
        self,
        intermediate_dim,
        num_heads,
        dropout=0,
        activation="relu",
        layer_norm_epsilon=1e-05,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        normalize_first=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation =tf.keras.activations.get(activation)
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer =tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer =tf.keras.initializers.get(bias_initializer)
        self.normalize_first = normalize_first
        self.supports_masking = True

    def build(self, inputs_shape):
        # Infer the dimension of our hidden feature size from the build shape.
        hidden_dim = inputs_shape[-1]
        # Attention head size is `hidden_dim` over the number of heads.
        key_dim = int(hidden_dim // self.num_heads)
        if key_dim == 0:
            raise ValueError(
                "Attention `key_dim` computed cannot be zero. "
                f"The `hidden_dim` value of {hidden_dim} has to be equal to "
                f"or greater than `num_heads` value of {self.num_heads}."
            )

        # Self attention layers.
        self._self_attention_layer = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=key_dim,
            dropout=self.dropout,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            dtype=self.dtype_policy,
            name="%s/self_attention_layer" % self.name,
        )
        if hasattr(self._self_attention_layer, "_build_from_signature"):
            self._self_attention_layer._build_from_signature(
                query=inputs_shape,
                value=inputs_shape,
            )
        else:
            self._self_attention_layer.build(
                query_shape=inputs_shape,
                value_shape=inputs_shape,
            )
        
        self._self_attention_layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="%s/self_attention_layer_norm" % self.name,
        )
        self._self_attention_dropout = tf.keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
            name="%s/self_attention_dropout" % self.name,
        )

        # Feedforward layers.
        self._feedforward_layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="%s/ff_norm" % self.name,
        )
        self._feedforward_intermediate_dense = tf.keras.layers.Dense(
            self.intermediate_dim,
            activation=self.activation,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            dtype=self.dtype_policy,
            name="%s/ff_intermediate" % self.name,
        )
        self._feedforward_output_dense = tf.keras.layers.Dense(
            hidden_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            dtype=self.dtype_policy,
            name="%s/FFdense" % self.name,
        )
        self._feedforward_dropout = tf.keras.layers.Dropout(
            rate=self.dropout,
            name="%s/feedforward_dropout" % self.name,
        )
        self.built = True

    def call(self, inputs, padding_mask=None, attention_mask=None):
        """Forward pass of the TransformerEncoder.

        Args:
            inputs: a Tensor. The input data to TransformerEncoder, should be
                of shape [batch_size, sequence_length, hidden_dim].
            padding_mask: a boolean Tensor. It indicates if the token should be
                masked because the token is introduced due to padding.
                `padding_mask` should have shape [batch_size, sequence_length].
            attention_mask: a boolean Tensor. Customized mask used to mask out
                certain tokens. `attention_mask` should have shape
                [batch_size, sequence_length, sequence_length].

        Returns:
            A Tensor of the same shape as the `inputs`.
        """
        x = inputs  # Intermediate result.

        # Compute self attention mask.
        self_attention_mask = merge_padding_and_attention_mask(
            inputs, padding_mask, attention_mask
        )

        # Self attention block.
        residual = x
        if self.normalize_first:
            x = self._self_attention_layer_norm(x)
        x = self._self_attention_layer(
            query=x,
            value=x,
            attention_mask=self_attention_mask,
        )
        x = self._self_attention_dropout(x)
        x = x + residual
        if not self.normalize_first:
            x = self._self_attention_layer_norm(x)

        # Feedforward block.
        residual = x
        if self.normalize_first:
            x = self._feedforward_layer_norm(x)
        x = self._feedforward_intermediate_dense(x)
        x = self._feedforward_output_dense(x)
        x = self._feedforward_dropout(x)
        x = x + residual
        if not self.normalize_first:
            x = self._feedforward_layer_norm(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "intermediate_dim": self.intermediate_dim,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
                "activation":tf.keras.activations.serialize(self.activation),
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "kernel_initializer":tf.keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer":tf.keras.initializers.serialize(
                    self.bias_initializer
                ),
                "normalize_first": self.normalize_first,
            }
        )
        return config

    def compute_output_shape(self, inputs_shape):
        return inputs_shape