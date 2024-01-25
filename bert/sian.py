from bert.bert_layer import BatchGather, DivConstant
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score
import math

keras = tf.keras
Dense = keras.layers.Dense
Embedding = keras.layers.Embedding
Add = keras.layers.Add
Concatenate = keras.layers.Concatenate
Lambda = keras.layers.Lambda
LayerNormalization = keras.layers.LayerNormalization
BatchNormalization = keras.layers.BatchNormalization
Dropout = keras.layers.Dropout
Softmax = keras.layers.Softmax
Reshape = keras.layers.Reshape
Input = keras.Input
Model = keras.models.Model
Multiply = keras.layers.Multiply
variable_scope = tf.compat.v1.variable_scope
truncated_normal = tf.compat.v1.truncated_normal_initializer

MODE_BHV, MODE_CATE, MODE_ATTR = 'bhv', 'cate', 'attr'


class DisAttLayer1(tf.keras.layers.Layer):
    def __init__(self, seq_len, n_mb, h, n_c=None, n_attr=None, **kwargs):  # 支持3种模式：bhv, bhv+cate(id), bhv+attr(multi-hot)
        super(DisAttLayer, self).__init__(**kwargs)
        self.seq_len = seq_len
        self.n_mb = n_mb
        self.h = h
        self.n_c = n_c
        self.n_attr = n_attr
        if n_c is None and n_attr is None:
            self.mode = MODE_BHV
        elif n_c is not None and n_attr is None:
            self.mode = MODE_CATE
        elif n_attr is not None:  # and n_c is None:
            self.mode = MODE_ATTR
        else:
            raise ValueError('n_c and n_attr not None')

    def build(self, input_shape):
        super(DisAttLayer, self).build(input_shape)
        a0, a1, a2, b0, p_dim, b_dim, ds = 64, 8, 4, 24, 16, 24, 24
        if self.mode in (MODE_CATE, MODE_ATTR):
            c_dim, ds = 24, 48
        self.e_pos = self.add_weight(shape=(self.seq_len*2, self.h, p_dim), initializer=truncated_normal(stddev=0.02), dtype=tf.float32, name='e_pos')
        self.e_bi = self.add_weight(shape=(self.n_mb+1, b_dim), initializer=truncated_normal(stddev=0.02), dtype=tf.float32, name='e_bi')
        self.e_bj = self.add_weight(shape=(self.n_mb+1, b_dim), initializer=truncated_normal(stddev=0.02), dtype=tf.float32, name='e_bj')
        # 类别特征
        if self.mode == MODE_CATE:
            self.e_ci = self.add_weight(shape=(self.n_c+2, c_dim), initializer=truncated_normal(stddev=0.02), dtype=tf.float32, name='e_ci')
            self.e_cj = self.add_weight(shape=(self.n_c+2, c_dim), initializer=truncated_normal(stddev=0.02), dtype=tf.float32, name='e_cj')
        if self.mode == MODE_ATTR:
            self.e_ai = self.add_weight(shape=(self.n_attr+1, c_dim), initializer=truncated_normal(stddev=0.02), dtype=tf.float32, name='e_ai')
            self.e_aj = self.add_weight(shape=(self.n_attr+1, c_dim), initializer=truncated_normal(stddev=0.02), dtype=tf.float32, name='e_aj')
        # 多头
        self.e_hi = self.add_weight(shape=(ds, self.h, b0), initializer=truncated_normal(stddev=0.02), dtype=tf.float32, name='e_hi')
        self.e_hj = self.add_weight(shape=(ds, self.h, b0), initializer=truncated_normal(stddev=0.02), dtype=tf.float32, name='e_hj')
        # dnn权重
        self.w1_e = self.add_weight(shape=(a0, a1, self.h), initializer=truncated_normal(stddev=0.02), dtype=tf.float32, name='w1_e')
        self.w2_e = self.add_weight(shape=(a1, a2, self.h), initializer=truncated_normal(stddev=0.02), dtype=tf.float32, name='w2_e')
        self.w3_e = self.add_weight(shape=(a2, self.h), initializer=truncated_normal(stddev=0.02), dtype=tf.float32, name='w3_e')

    def call(self, inputs, mask=None):
        b_seq, c_seq = inputs  # [b, n], [b, n]
        b_size = tf.shape(b_seq)[0]  # 128
        # 1. 提取相对位置特征
        q_pos = tf.range(self.seq_len, dtype=tf.int32)
        k_pos = tf.range(self.seq_len, dtype=tf.int32)
        f_pq = tf.tile(q_pos[None, :, None], [b_size, 1, self.seq_len])
        f_pk = tf.tile(k_pos[None, None, :], [b_size, self.seq_len, 1])
        f_pos = f_pq - f_pk + self.seq_len
        # 2. 事件类型特征
        if self.mode == MODE_ATTR:      # multi-hot sum pooling
            f_ci = tf.matmul(c_seq, self.e_ai)
            f_cj = tf.matmul(c_seq, self.e_aj)
        elif self.mode == MODE_CATE:
            f_ci = tf.nn.embedding_lookup(self.e_ci, c_seq)  # [b, n, C]
            f_cj = tf.nn.embedding_lookup(self.e_cj, c_seq)
        if self.mode == MODE_BHV:
            f_i = tf.nn.embedding_lookup(self.e_bi, b_seq)  # [b, n, B]
            f_j = tf.nn.embedding_lookup(self.e_bj, b_seq)
        elif self.mode in (MODE_CATE, MODE_ATTR):
            f_bi = tf.nn.embedding_lookup(self.e_bi, b_seq)  # [b, n, B]
            f_bj = tf.nn.embedding_lookup(self.e_bj, b_seq)
            f_i = tf.concat([f_bi, f_ci], axis=-1)
            f_j = tf.concat([f_bj, f_cj], axis=-1)
        # 3. 多头机制
        f_i = tf.einsum('bnd,dha->bnha', f_i, self.e_hi)  # [b, n, B+C] -> [b, n, h, b0]
        f_j = tf.einsum('bnd,dha->bnha', f_j, self.e_hj)
        f_i = tf.tile(f_i[:, :, None, :], [1, 1, self.seq_len, 1, 1])  # [b, n, n, h, b0]
        f_j = tf.tile(f_j[:, None, :, :], [1, self.seq_len, 1, 1, 1])

        f_emb = [tf.nn.embedding_lookup(self.e_pos, f_pos), f_i, f_j]
        e = tf.concat(f_emb, axis=-1)
        e_w1 = tf.einsum('bmnhd,dkh->bmnhk', e, self.w1_e)
        e_w1_relu = tf.nn.relu(e_w1)
        e_w2 = tf.einsum('bmnhk,klh->bmnhl', e_w1_relu, self.w2_e)
        e_w2_relu = tf.nn.relu(e_w2)
        da_score = tf.einsum('bmnhk,kh->bhmn', e_w2_relu, self.w3_e)
        return da_score

    def compute_output_shape(self, input_shape=None):
        return input_shape[0]

    def get_config(self):
        config = {'seq_len': self.seq_len, 'n_mb': self.n_mb, 'h': self.h}
        base_config = super(DisAttLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DisAttLayer(tf.keras.layers.Layer):
    def __init__(self, seq_len, n_mb, h, n_c=None, n_attr=None, **kwargs):
        super(DisAttLayer, self).__init__(**kwargs)
        self.seq_len = seq_len
        self.n_mb = n_mb
        self.h = h
        self.n_c = n_c
        self.n_attr = n_attr

    def build(self, input_shape):
        # assert isinstance(input_shape, list)
        super(DisAttLayer, self).build(input_shape)
        # p_size, b_size, l0_size, l1_size, l2_size = 32, 16, 64, 32, 16  # 64, 32, 96, 64, 32
        if self.n_c is None:
            p_size, b_size, c_size, l0_size, l1_size, l2_size = 32, 16, 16, 64, 32, 16
        elif self.n_attr is not None:
            p_size, b_size, c_size, l0_size, l1_size, l2_size = 32, 16, 16, 96, 32, 16
            self.w_attr_1 = self.add_weight(shape=(self.n_attr+1, c_size), initializer=truncated_normal(stddev=0.02), dtype=tf.float32, name='w_attr_1')
            self.w_attr_2 = self.add_weight(shape=(self.n_attr + 1, c_size), initializer=truncated_normal(stddev=0.02),
                                            dtype=tf.float32, name='w_attr_2')
        else:
            p_size, b_size, c_size, l0_size, l1_size, l2_size = 32, 16, 16, 96, 32, 16
        self.e_pos = self.add_weight(shape=(self.seq_len*2, self.h, p_size), initializer=truncated_normal(stddev=0.02), dtype=tf.float32, name='e_pos')
        self.e_bi = self.add_weight(shape=(self.n_mb+1, self.h, b_size), initializer=truncated_normal(stddev=0.02), dtype=tf.float32, name='e_bi')
        self.e_bj = self.add_weight(shape=(self.n_mb+1, self.h, b_size), initializer=truncated_normal(stddev=0.02), dtype=tf.float32, name='e_bj')
        # 类别特征
        if self.n_c is not None and self.n_attr is None:
            self.e_ci = self.add_weight(shape=(self.n_c+2, self.h, c_size), initializer=truncated_normal(stddev=0.02), dtype=tf.float32, name='e_ci')
            self.e_cj = self.add_weight(shape=(self.n_c+2, self.h, c_size), initializer=truncated_normal(stddev=0.02), dtype=tf.float32, name='e_cj')
        # dnn权重
        self.w1_e = self.add_weight(shape=(l0_size, l1_size, self.h), initializer=truncated_normal(stddev=0.02), dtype=tf.float32, name='w1_e')
        self.w2_e = self.add_weight(shape=(l1_size, l2_size, self.h), initializer=truncated_normal(stddev=0.02), dtype=tf.float32, name='w2_e')
        self.w3_e = self.add_weight(shape=(l2_size, self.h), initializer=truncated_normal(stddev=0.02), dtype=tf.float32, name='w3_e')

    def call(self, inputs, mask=None):
        b_seq, c_seq = inputs  # [b, h, n, k]
        b_size = tf.shape(b_seq)[0]  # 128
        # 1. 提取相对位置特征
        q_pos = tf.range(self.seq_len, dtype=tf.int32)
        k_pos = tf.range(self.seq_len, dtype=tf.int32)
        f_pq = tf.tile(q_pos[None, :, None], [b_size, 1, self.seq_len])
        f_pk = tf.tile(k_pos[None, None, :], [b_size, self.seq_len, 1])
        f_pos = f_pq - f_pk + self.seq_len
        # 2. 事件类型特征
        f_bi = tf.tile(b_seq[:, :, None], [1, 1, self.seq_len])
        f_bj = tf.tile(b_seq[:, None, :], [1, self.seq_len, 1])
        # 3. 类别特征
        if self.n_c is not None and c_seq is not None:
            if self.n_attr is None:
                f_ci = tf.tile(c_seq[:, :, None], [1, 1, self.seq_len])
                f_cj = tf.tile(c_seq[:, None, :], [1, self.seq_len, 1])
            else:
                c_seq_i = tf.matmul(c_seq, self.w_attr_1)
                f_ci = tf.tile(c_seq_i[:, :, None, None, :], [1, 1, self.seq_len, self.h, 1])
                c_seq_j = tf.matmul(c_seq, self.w_attr_2)
                f_cj = tf.tile(c_seq_j[:, None, :, None, :], [1, self.seq_len, 1, self.h, 1])
        # embedding concat
        if self.n_c is None:
            f_emb = [tf.nn.embedding_lookup(self.e_pos, f_pos), tf.nn.embedding_lookup(self.e_bi, f_bi),
                     tf.nn.embedding_lookup(self.e_bj, f_bj)]
        else:
            if self.n_attr is None:
                f_emb = [tf.nn.embedding_lookup(self.e_pos, f_pos), tf.nn.embedding_lookup(self.e_bi, f_bi), tf.nn.embedding_lookup(self.e_bj, f_bj),
                         tf.nn.embedding_lookup(self.e_ci, f_ci), tf.nn.embedding_lookup(self.e_cj, f_cj)]
            else:
                f_emb = [tf.nn.embedding_lookup(self.e_pos, f_pos), tf.nn.embedding_lookup(self.e_bi, f_bi),
                         tf.nn.embedding_lookup(self.e_bj, f_bj),
                         f_ci, f_cj]
        e = tf.concat(f_emb, axis=-1)
        e_w1 = tf.einsum('bmnhd,dkh->bmnhk', e, self.w1_e)
        e_w1_relu = tf.nn.relu(e_w1)
        e_w2 = tf.einsum('bmnhk,klh->bmnhl', e_w1_relu, self.w2_e)
        e_w2_relu = tf.nn.relu(e_w2)
        da_score = tf.einsum('bmnhk,kh->bhmn', e_w2_relu, self.w3_e)
        score = da_score
        return score

    def compute_output_shape(self, input_shape=None):
        return input_shape[0]

    def get_config(self):
        config = {'seq_len': self.seq_len, 'n_mb': self.n_mb, 'h': self.h}
        base_config = super(DisAttLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == "__main__":
    seq_len, n_mb, h = 4, 4, 2
    # 1. CATE
    # da = DisAttLayer(seq_len, n_mb, h, n_c=4, n_attr=None)
    # da.build(input_shape=None)
    # bs, cs = tf.constant([[1, 4, 2, 1], [1, 2, 2, 3], [1, 4, 2, 3]]), tf.constant([[1, 4, 2, 1], [1, 2, 2, 3], [1, 4, 2, 3]])
    # out1 = da.call([bs, cs])
    # 2. ATTR
    da = DisAttLayer(seq_len, n_mb, h, n_attr=1)
    da.build(input_shape=None)
    bs, cs = tf.constant([[1, 4, 2, 1], [1, 2, 2, 3], [1, 4, 2, 3]]), tf.constant(
        [[[0, 1], [1, 0], [1, 1], [0, 0]], [[0, 1], [1, 0], [1, 1], [0, 0]], [[0, 1], [1, 0], [1, 1], [0, 0]]], dtype=tf.float32)
    out2 = da.call([bs, cs])
