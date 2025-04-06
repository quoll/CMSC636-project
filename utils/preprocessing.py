import tensorflow as tf
from config.constants import bounding_square

def pad_to_fixed_size(image, target_size=(bounding_square, bounding_square)):
    current_height = tf.shape(image)[0]
    current_width = tf.shape(image)[1]

    scale = tf.minimum(
        target_size[0] / tf.cast(current_height, tf.float32),
        target_size[1] / tf.cast(current_width, tf.float32)
    )

    def resize_needed():
        new_height = tf.cast(tf.cast(current_height, tf.float32) * scale, tf.int32)
        new_width = tf.cast(tf.cast(current_width, tf.float32) * scale, tf.int32)
        return tf.image.resize(image, [new_height, new_width], method='bilinear')

    def no_resize_needed():
        return image

    image = tf.cond(
        tf.logical_or(current_height > target_size[0], current_width > target_size[1]),
        resize_needed,
        no_resize_needed
    )

    current_height = tf.shape(image)[0]
    current_width = tf.shape(image)[1]

    pad_height = target_size[0] - current_height
    pad_width = target_size[1] - current_width

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    padded = tf.pad(
        image,
        paddings=[[pad_top, pad_bottom], [pad_left, pad_right], [0,0]],
        mode='CONSTANT',
        constant_values=0
    )
    padded.set_shape([bounding_square, bounding_square, 1])
    return padded
