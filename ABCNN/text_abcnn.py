# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import fully_connected, xavier_initializer, l2_regularizer, softmax
import keras.backend as K
from keras.layers import Dense, merge, TimeDistributed, Permute, Reshape


def linear(input_, output_size, scope=None):
    """
    Linear map: output[k] = sum_i(Matrix[k, i] * args[i] ) + Bias[k]
    Args:
        input_: a tensor or a list of 2D, batch x n, Tensors.
        output_size: int, second dimension of W[i].
        scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
    """

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: {0}".format(str(shape)))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: {0}".format(str(shape)))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "SimpleLinear"):
        W = tf.get_variable("W", [output_size, input_size], dtype=input_.dtype)
        b = tf.get_variable("b", [output_size], dtype=input_.dtype)

    return tf.nn.xw_plus_b(input_, tf.transpose(W), b)


def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """
    Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope=('highway_lin_{0}'.format(idx))))
            t = tf.sigmoid(linear(input_, size, scope=('highway_gate_{0}'.format(idx))) + bias)
            output = t * g + (1. - t) * input_
            input_ = output

    return output


class TextABCNN(object):
    """A ABCNN for text classification."""

    def __init__(
            self, sequence_length, num_classes, model_type, vocab_size, fc_hidden_size, embedding_size,
            embedding_type, filter_sizes, num_filters, l2_reg_lambda=0.0, pretrained_embedding=None):

        # Placeholders for input, output, dropout_prob and training_tag
        self.input_x_front = tf.placeholder(tf.int32, [None, sequence_length], name="input_x_front")
        self.input_x_behind = tf.placeholder(tf.int32, [None, sequence_length], name="input_x_behind")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")

        def cos_sim(input_x1, input_x2):
            norm1 = tf.square(tf.reduce_sum(tf.square(input_x1), axis=1))
            norm2 = tf.square(tf.reduce_sum(tf.square(input_x2), axis=1))
            dot_products = tf.reduce_sum(input_x1 * input_x2, axis=1, name="cos_sim")
            return dot_products / (norm1 * norm2)

        def make_attention_mat(input_x1, input_x2):
            # shape of `input_x1` and `input_x2`: [batch_size, embedding_size, sequence_length, 1]
            # input_x2 need to transpose to the [batch_size, embedding_size, 1, sequence_length]
            # shape of output: [batch_size, sequence_length, sequence_length]
            euclidean = tf.sqrt(tf.reduce_sum(tf.square(input_x1 - tf.matrix_transpose(input_x2)), axis=1))
            return 1 / (1 + euclidean)

        def w_pool(input_x, attention, filter_size, scope):
            # input_x: [batch_size, num_filters, sequence_length + filter_size - 1, 1]
            # attention: [batch_size, sequence_length + filter_size - 1]
            if model_type in ['ABCNN2', 'ABCNN3']:
                pools = []

                # [batch_size, 1, sequence_length + filter_size - 1, 1]
                attention = tf.transpose(tf.expand_dims(tf.expand_dims(attention, -1), -1), [0, 2, 1, 3])

                for i in range(sequence_length):
                    # [batch_size, num_filters, filter_size, 1]
                    # reduce_sum => [batch_size, num_filters, 1, 1]
                    pools.append(
                        tf.reduce_sum(input_x[:, :, i:i + filter_size, :] * attention[:, :, i:i + filter_size, :],
                                      axis=2, keepdims=True))
                # [batch_size, num_filters, sequence_length, 1]
                w_ap = tf.concat(pools, axis=2, name="w_ap_" + scope)
            else:
                # [batch_size, num_filters, sequence_length, 1]
                w_ap = tf.nn.avg_pool(
                    input_x,
                    ksize=[1, 1, filter_size, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="w_ap_" + scope
                )
            return w_ap

        def all_pool(input_x, filter_size, scope):
            # input_x: [batch_size, num_filters, sequence_length + filter_size -1, 1]
            all_ap = tf.nn.avg_pool(
                input_x,
                ksize=[1, 1, sequence_length + filter_size - 1, 1],
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="all_pool_" + scope
            )
            all_ap_reshaped = tf.reshape(all_ap, [-1, num_filters])
            return all_ap_reshaped

        def cnn_layer(variable_scope, input_x1, input_x2, dims):
            """
            Args:
                variable_scope: `cnn-1` or `cnn-2`
                input_x1:
                dims: embedding_size in `cnn-1`, num_filters in `cnn-2`
            """

            with tf.name_scope(variable_scope):
                if model_type in ['ABCNN1', 'ABCNN3']:
                    # Attention
                    with tf.name_scope("attention_matrix"):
                        W_a = tf.Variable(tf.truncated_normal(shape=[sequence_length, embedding_size],
                                                              stddev=0.1, dtype=tf.float32), name="W_a")
                        # shape of `attention_matrix`: [batch_size, sequence_length, sequence_length]
                        attention_matrix = make_attention_mat(self.embedded_sentence_expanded_front_trans,
                                                              self.embedded_sentence_expanded_behind_trans)

                        # [batch_size, sequence_length, sequence_length] * [sequence_length, embedding_size]
                        # einsum => [batch_size, sequence_length, embedding_size]
                        # matrix transpose => [batch_size, embedding_size, sequence_length]
                        # expand dims => [batch_size, embedding_size, sequence_length, 1]
                        front_attention = tf.expand_dims(
                            tf.matrix_transpose(tf.einsum("ijk,kl->ijl", attention_matrix, W_a)), -1)
                        behind_attention = tf.expand_dims(
                            tf.matrix_transpose(tf.einsum("ijk,kl->ijl", tf.matrix_transpose(attention_matrix), W_a)), -1)

                        # shape of new `embedded_sentence_expanded_front`: [batch_size, sequence_length, embedding_size, 2]
                        self.embedded_sentence_expanded_front = tf.transpose(tf.concat(
                            [self.embedded_sentence_expanded_front_trans, front_attention], axis=3), perm=[0, 2, 1, 3])
                        self.embedded_sentence_expanded_behind = tf.transpose(tf.concat(
                            [self.embedded_sentence_expanded_behind_trans, behind_attention], axis=3), perm=[0, 2, 1, 3])

                for filter_size in filter_sizes:
                    with tf.name_scope("conv-filter{0}".format(filter_size)):
                        # Convolution Layer
                        if model_type in ['ABCNN1', 'ABCNN3']:
                            # The in_channels of filter_shape is 2 (two channels, origin + attention)
                            in_channels = 2
                        else:
                            in_channels = 1

                        # shape of new `embedded_sentence_expanded_front`
                        # [batch_size, sequence_length + filter_size - 1, embedding_size, 2]
                        self.embedded_sentence_expanded_front = tf.pad(self.embedded_sentence_expanded_front, np.array(
                            [[0, 0], [filter_size - 1, filter_size - 1], [0, 0], [0, 0]]), "CONSTANT")
                        self.embedded_sentence_expanded_behind = tf.pad(self.embedded_sentence_expanded_behind, np.array(
                            [[0, 0], [filter_size - 1, filter_size - 1], [0, 0], [0, 0]]), "CONSTANT")

                        filter_shape = [filter_size, embedding_size, in_channels, num_filters]
                        W = tf.Variable(tf.truncated_normal(shape=filter_shape, stddev=0.1, dtype=tf.float32), name="W")
                        b = tf.Variable(tf.constant(0.1, shape=[num_filters], dtype=tf.float32), name="b")
                        conv_front = tf.nn.conv2d(
                            self.embedded_sentence_expanded_front,
                            W,
                            strides=[1, 1, 1, 1],
                            padding="VALID",
                            name="conv_front")

                        conv_behind = tf.nn.conv2d(
                            self.embedded_sentence_expanded_behind,
                            W,
                            strides=[1, 1, 1, 1],
                            padding="VALID",
                            name="conv_behind")

                        # Batch Normalization Layer
                        conv_bn_front = tf.layers.batch_normalization(
                            tf.nn.bias_add(conv_front, b), training=self.is_training)
                        conv_bn_behind = tf.layers.batch_normalization(
                            tf.nn.bias_add(conv_behind, b), training=self.is_training)

                        # Apply nonlinearity
                        # [batch_size, sequence_length + filter_size - 1, 1, num_filters]
                        conv_out_front = tf.nn.relu(conv_bn_front, name="relu_front")
                        conv_out_behind = tf.nn.relu(conv_bn_behind, name="relu_behind")

                        # [batch_size, num_filters, sequence_length + filter_size - 1, 1]
                        conv_out_front_trans = tf.transpose(conv_out_front, perm=[0, 3, 1, 2])
                        conv_out_behind_trans = tf.transpose(conv_out_behind, perm=[0, 3, 1, 2])

                    front_attention_v2, behind_attention_v2 = None, None

                    if model_type in ['ABCNN2', 'ABCNN3']:
                        # [batch_size, sequence_length + filter_size - 1, sequence_length + filter_size - 1]
                        attention_matrix_v2 = make_attention_mat(conv_out_front_trans, conv_out_behind_trans)

                        # [batch_size, sequence_length + filter_size - 1]
                        front_attention_v2 = tf.reduce_sum(attention_matrix_v2, axis=2)
                        behind_attention_v2 = tf.reduce_sum(attention_matrix_v2, axis=1)

                    with tf.name_scope("pool-filter{0}".format(filter_size)):
                        # shape of `front_wp`: [batch_size, num_filters, sequence_length, 1]
                        front_wp = w_pool(input_x=conv_out_front_trans, attention=front_attention_v2,
                                          filter_size=filter_size, scope="front")
                        behind_wp = w_pool(input_x=conv_out_behind_trans, attention=behind_attention_v2,
                                           filter_size=filter_size, scope="behind")

                        # shape of `front_ap`: [batch_size, num_filters]
                        front_ap = all_pool(input_x=conv_out_front_trans, filter_size=filter_size, scope="front")
                        behind_ap = all_pool(input_x=conv_out_behind_trans, filter_size=filter_size, scope="behind")

                        FI_1, BI_1 = front_wp, behind_wp
                        F0_1, B0_1 = front_ap, behind_ap
                    


        # Embedding Layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # Use random generated the word vector by default
            # Can also be obtained through our own word vectors trained by our corpus
            if pretrained_embedding is None:
                self.embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0,
                                                               dtype=tf.float32), trainable=True, name="embedding")
            else:
                if embedding_type == 0:
                    self.embedding = tf.constant(pretrained_embedding, dtype=tf.float32, name="embedding")
                if embedding_type == 1:
                    self.embedding = tf.Variable(pretrained_embedding, trainable=True,
                                                 dtype=tf.float32, name="embedding")
            self.embedded_sentence_front = tf.nn.embedding_lookup(self.embedding, self.input_x_front)
            self.embedded_sentence_behind = tf.nn.embedding_lookup(self.embedding, self.input_x_behind)
            self.embedded_sentence_expanded_front = tf.expand_dims(self.embedded_sentence_front, -1)
            self.embedded_sentence_expanded_behind = tf.expand_dims(self.embedded_sentence_behind, -1)

        self.embedded_sentence_front_trans = tf.transpose(self.embedded_sentence_front, perm=[0, 2, 1])
        self.embedded_sentence_behind_trans = tf.transpose(self.embedded_sentence_behind, perm=[0, 2, 1])

        # [batch_size, embedding_size, sequence_length, 1]
        self.embedded_sentence_expanded_front_trans = tf.expand_dims(self.embedded_sentence_front_trans, -1)
        self.embedded_sentence_expanded_behind_trans = tf.expand_dims(self.embedded_sentence_behind_trans, -1)

        # Average-pooling Layer
        with tf.name_scope("input-all-avg_pool"):
            self.embedded_sentence_front_avg_pool = tf.nn.avg_pool(
                self.embedded_sentence_expanded_front,
                ksize=[sequence_length, 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="all_avg_pool_front"
            )

            self.embedded_sentence_behind_avg_pool = tf.nn.avg_pool(
                self.embedded_sentence_expanded_behind,
                ksize=[sequence_length, 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="all_avg_pool_behind"
            )
            # shape of `L0_0` and `R0_0`: [batch_size, embedding_size]
            self.F0_0 = tf.reshape(self.embedded_sentence_front_avg_pool, [-1, embedding_size])
            self.B0_0 = tf.reshape(self.embedded_sentence_behind_avg_pool, [-1, embedding_size])

        pooled_outputs_front = []
        pooled_outputs_behind = []
        sims = []

        FI_1, F0_1, BI_1, B0_1 = cnn_layer(variable_scope="CNN-1", x1=x1_expanded, x2=x2_expanded, d=d0)

        pooled_outputs_front.append(F0_1)
        pooled_outputs_behind.append(B0_1)

        # Convolution Layer

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.pool_front = tf.concat(pooled_outputs_front, 1)
        self.pool_behind = tf.concat(pooled_outputs_behind, 1)
        # self.pool_flat_combine = tf.concat([self.pool_flat_front, self.pool_flat_behind], 1)

        sims.append([cos_sim(self.F0_0, self.B0_0), cos_sim(self.pool_front, self.pool_behind)])
        self.pool_features = tf.transpose(tf.stack(sims, axis=1), perm=[2, 0, 1])
        print(self.pool_features)

        self.fc_out = fully_connected(
            inputs=self.pool_features,
            num_outputs=fc_hidden_size,
            activation_fn=tf.nn.relu,
            weights_initializer=xavier_initializer(),
            weights_regularizer=l2_regularizer(scale=l2_reg_lambda),
            biases_initializer=tf.constant_initializer(0.1),
            scope="fc"
        )

        print(self.fc_out)
        self.haha = softmax(self.fc_out)[:, 1]
        print(self.haha)

        # Fully Connected Layer
        with tf.name_scope("fc"):
            W = tf.Variable(tf.truncated_normal(shape=[num_filters_total * 2, fc_hidden_size],
                                                stddev=0.1, dtype=tf.float32), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[fc_hidden_size], dtype=tf.float32), name="b")
            self.fc = tf.nn.xw_plus_b(self.pool_flat_combine, W, b)

            # Batch Normalization Layer
            self.fc_bn = tf.layers.batch_normalization(self.fc, training=self.is_training)

            # Apply nonlinearity
            self.fc_out = tf.nn.relu(self.fc_bn, name="relu")

        # Highway Layer
        self.highway = highway(self.fc_out, self.fc_out.get_shape()[1], num_layers=1, bias=0, scope="Highway")

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.highway, self.dropout_keep_prob)

        # Final scores and predictions
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal(shape=[fc_hidden_size, num_classes],
                                                stddev=0.1, dtype=tf.float32), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes], dtype=tf.float32), name="b")
            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="logits")
            self.softmax_scores = tf.nn.softmax(self.logits, name="softmax_scores")
            self.predictions = tf.argmax(self.logits, 1, name="predictions")
            self.topKPreds = tf.nn.top_k(self.softmax_scores, k=1, sorted=True, name="topKPreds")

        # Calculate mean cross-entropy loss, L2 loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.input_y, logits=self.logits)
            losses = tf.reduce_mean(losses, name="softmax_losses")
            l2_losses = tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()],
                                 name="l2_losses") * l2_reg_lambda
            self.loss = tf.add(losses, l2_losses, name="loss")

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        # Number of correct predictions
        with tf.name_scope("num_correct"):
            correct = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.num_correct = tf.reduce_sum(tf.cast(correct, "float"), name="num_correct")

        # Calculate Fp
        with tf.name_scope("fp"):
            fp = tf.metrics.false_positives(labels=tf.argmax(self.input_y, 1), predictions=self.predictions)
            self.fp = tf.reduce_sum(tf.cast(fp, "float"), name="fp")

        # Calculate Fn
        with tf.name_scope("fn"):
            fn = tf.metrics.false_negatives(labels=tf.argmax(self.input_y, 1), predictions=self.predictions)
            self.fn = tf.reduce_sum(tf.cast(fn, "float"), name="fn")

        # Calculate Recall
        with tf.name_scope("recall"):
            self.recall = self.num_correct / (self.num_correct + self.fn)

        # Calculate Precision
        with tf.name_scope("precision"):
            self.precision = self.num_correct / (self.num_correct + self.fp)

        # Calculate F1
        with tf.name_scope("F1"):
            self.F1 = (2 * self.precision * self.recall) / (self.precision + self.recall)

        # Calculate AUC
        with tf.name_scope("AUC"):
            self.AUC = tf.metrics.auc(self.softmax_scores, self.input_y, name="AUC")
