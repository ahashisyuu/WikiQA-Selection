import tensorflow as tf


def Dropout(args, keep_prob, is_train, mode="recurrent", name=None):

    def _dropout():
        _args = args
        noise_shape = None
        scale = 1.0
        shape = tf.shape(_args)
        if mode == "embedding":
            noise_shape = [shape[0], 1]
            scale = keep_prob
        if mode == "recurrent" and len(_args.get_shape().as_list()) == 3:
            noise_shape = [shape[0], 1, shape[-1]]
        _args = tf.cond(is_train, lambda: tf.nn.dropout(
            _args, keep_prob, noise_shape=noise_shape, name=name) * scale, lambda: _args)
        return _args

    return tf.cond(tf.less(keep_prob, 1.0), _dropout, lambda: args)