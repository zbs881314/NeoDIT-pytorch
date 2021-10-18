import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from tflearn.initializations import truncated_normal
from tflearn.activations import relu


def weight_variable(shape):
    """
    权重变量
    :param shape:
    :return:
    """
    # truncated_normal -> 截尾正态分布
    # stddev -> 截断正态分布的标准差
    initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
    # tf.Variable()说明
    # 创建一个值initial_value的新变量。
    # 新变量被添加到collections中列出的图集合中，默认值为[GraphKeys.GLOBAL_VARIABLES]。
    # 如果trainable为True，则该变量也会添加到图形集合GraphKeys.TRAINABLE_VARIABLES中。
    # 这个构造函数创建变量Op和赋值Op，以将变量设置为初始值。
    return tf.Variable(initial, dtype=tf.float32)


def bias_variable(shape):
    """
    偏差变量
    :param shape:
    :return:
    """
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(initial, dtype=tf.float32)


def a_layer(x, units):
    """
    单层
    :param x:
    :param units:
    :return:
    """
    W = weight_variable([x.get_shape().as_list()[1], units])
    b = bias_variable([units])
    tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W))
    return relu(tf.matmul(x, W) + b)


def bi_layer(x0, x1, sym, dim_pred):
    """
    双层
    :param x0:
    :param x1:
    :param sym:
    :param dim_pred:项目矩阵维度
    :return:
    """
    if not sym:
        W0p = weight_variable([x0.get_shape().as_list()[1], dim_pred])
        W1p = weight_variable([x1.get_shape().as_list()[1], dim_pred])
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W0p))
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W1p))
        return tf.matmul(tf.matmul(x0, W0p),
                         tf.matmul(x1, W1p), transpose_b=True)
    else:
        W0p = weight_variable([x0.get_shape().as_list()[1], dim_pred])
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W0p))
        return tf.matmul(tf.matmul(x0, W0p),
                         tf.matmul(x1, W0p), transpose_b=True)
