import tensorflow as tf

class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = int(patch_size)

    def call(self, inputs):
        # Compute sizes for reshaping the patches later
        batch_size, height, width, channels = [tf.shape(inputs)[i] for i in range(4)]
        # Ensure the patch size is compatible for both dimensions
        # assert ((height % self.patch_size) == 0) and ((width % self.patch_size) == 0), \
        #     "Image dimensions must be divisible by the patch size."

        # Extract patches using tf.image.extract_patches
        patches = tf.image.extract_patches(
          images=inputs,
          sizes=[1, self.patch_size, self.patch_size, 1],
          strides=[1, self.patch_size, self.patch_size, 1],
          rates=[1, 1, 1, 1],
          padding='VALID'
        )

        # Calculate the dimensions of the patches
        patch_dim = patches.shape[-1]
        patch_height = height // self.patch_size
        patch_width = width // self.patch_size

        # Reshape to have the patch dimension first
        patches = tf.reshape(patches, [batch_size, patch_height * patch_width, patch_dim])
        return patches
