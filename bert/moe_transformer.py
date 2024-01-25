from .bert_layer import BatchGather, DivConstant
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score
import math
from .sian import DisAttLayer as Sian

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


class PeLayer(tf.keras.layers.Layer):
    def __init__(self, seq_len, n_mb, h, num_buckets=32, max_distance=128, **kwargs):
        super(PeLayer, self).__init__(**kwargs)
        self.seq_len = seq_len
        self.n_mb = n_mb
        self.h = h
        self.num_buckets = num_buckets
        self.max_distance = max_distance

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        super(PeLayer, self).build(input_shape)
        self.W1 = self.add_weight(shape=(self.num_buckets, self.n_mb*self.n_mb + 1, self.h), initializer=keras.initializers.TruncatedNormal(stddev=0.02), dtype=tf.float32, name='mb_pe_w1')

    @staticmethod
    def relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position
        num_buckets //= 2
        ret += tf.cast(n < 0, tf.int32) * num_buckets  # mtf.to_int32(mtf.less(n, 0)) * num_buckets
        n = tf.abs(n)
        # now n is in the range [0, inf)
        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        val_if_large = max_exact + tf.cast(
            tf.math.log(n / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact), tf.int32)
        val_if_large = tf.math.minimum(val_if_large, tf.experimental.numpy.full_like(val_if_large, num_buckets - 1))
        ret += tf.where(is_small, n, val_if_large)
        return ret

    def call(self, inputs, mask=None):
        attention_scores, b_seq = inputs  # [b, h, n, k]
        q_pos = tf.range(self.seq_len, dtype=tf.int32)
        k_pos = tf.range(self.seq_len, dtype=tf.int32)
        relative_position = k_pos[None, :] - q_pos[:, None]
        """
                   k
             0   1   2   3
        q   -1   0   1   2
            -2  -1   0   1
            -3  -2  -1   0
        """
        rp_bucket = self.relative_position_bucket(
            relative_position, num_buckets=self.num_buckets, max_distance=self.max_distance)
        rp_bucket = tf.expand_dims(rp_bucket, 0)
        pe = tf.nn.embedding_lookup(self.W1, rp_bucket)     # [1, n, n, C, h]
        # pe = tf.transpose(pe, [0, 4, 1, 2, 3])
        # b_mat = ((b_seq[:, :, None] - 1) * self.n_mb + b_seq[:, None, :]) * tf.cast(
        #     b_seq[:, :, None] * b_seq[:, None, :] != 0, tf.int32)
        # ind = tf.tile(b_mat[:, None, :, :, None], (1, self.h, 1, 1, 1))
        # pe_mb = tf.gather_nd(pe, ind, batch_dims=4)

        b_mat = ((b_seq[:, :, None] - 1) * self.n_mb + b_seq[:, None, :]) * tf.cast(
            b_seq[:, :, None] * b_seq[:, None, :] != 0, tf.int32)
        one_hot_b_mat = tf.tile(tf.one_hot(b_mat[:, None, :, :], depth=self.n_mb * self.n_mb + 1), (1, self.h, 1, 1, 1))
        pe_mb = tf.einsum('lmnCh,bhmnC->bhmn', pe, one_hot_b_mat)
        score = attention_scores + pe_mb
        return score

    def compute_output_shape(self, input_shape=None):
        return [self.seq_len, self.seq_len]

    def get_config(self):
        config = {'seq_len': self.seq_len, 'n_mb': self.n_mb, 'num_buckets': self.num_buckets, 'max_distance': self.max_distance, 'h': self.h}
        base_config = super(PeLayer, self).get_config()
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
        attention_scores, b_seq, c_seq = inputs  # [b, h, n, k]
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
        score = attention_scores + da_score
        return score

    def compute_output_shape(self, input_shape=None):
        return input_shape[0]

    def get_config(self):
        config = {'seq_len': self.seq_len, 'n_mb': self.n_mb, 'h': self.h}
        base_config = super(DisAttLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MbHeadLayer(tf.keras.layers.Layer):
    def __init__(self, n_mb, h, k, **kwargs):
        super(MbHeadLayer, self).__init__(**kwargs)
        self.n_mb = n_mb
        self.h = h
        self.k = k

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        super(MbHeadLayer, self).build(input_shape)
        self.W1 = self.add_weight(shape=(self.n_mb, self.h, self.k, self.k), initializer=keras.initializers.TruncatedNormal(stddev=0.02), dtype=tf.float32, name='mb_head_w1')
        self.alpha1 = self.add_weight(shape=(self.n_mb * self.n_mb + 1, self.n_mb, self.h), initializer=keras.initializers.TruncatedNormal(stddev=0.02), dtype=tf.float32, name='mb_head_alpha1')

    def call(self, inputs, mask=None):
        assert isinstance(inputs, list)
        query, key, b_seq = inputs  # [b, h, n, d], [b, n]
        b_mat = ((b_seq[:,:,None]-1)*self.n_mb + b_seq[:,None,:]) * tf.cast(b_seq[:,:,None]*b_seq[:,None,:]!=0, tf.int32)
        W1_ = tf.einsum('Bhmn,CBh->Chmn', self.W1, tf.nn.softmax(self.alpha1, 1))
        att_all = tf.einsum('bhim,Chmn,bhjn->bhijC', query, W1_, key)
        ind = tf.tile(b_mat[:, None, :, :, None], (1, self.h, 1, 1, 1))
        scores = tf.gather_nd(att_all, ind, batch_dims=4)
        return scores

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        print(input_shape[-1])
        return input_shape[-1]+input_shape[-1:]

    def get_config(self):
        config = {'n_mb': self.n_mb, 'h': self.h, 'k': self.k}
        base_config = super(MbHeadLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MbValueLayer(tf.keras.layers.Layer):
    def __init__(self, n_mb, h, k, **kwargs):
        super(MbValueLayer, self).__init__(**kwargs)
        self.n_mb = n_mb
        self.h = h
        self.k = k

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        super(MbValueLayer, self).build(input_shape)
        self.W2 = self.add_weight(shape=(self.n_mb, self.h, self.k, self.k), initializer=keras.initializers.TruncatedNormal(stddev=0.02), dtype=tf.float32, name='mb_head_w2')
        self.alpha2 = self.add_weight(shape=(self.n_mb * self.n_mb + 1, self.n_mb, self.h), initializer=keras.initializers.TruncatedNormal(stddev=0.02), dtype=tf.float32, name='mb_head_alpha2')

    def call(self, inputs, mask=None):
        assert isinstance(inputs, list)
        p_attn, value, b_seq = inputs  # [b, n, d], [b, n]
        b_mat = ((b_seq[:,:,None]-1)*self.n_mb + b_seq[:,None,:]) * tf.cast(b_seq[:,:,None]*b_seq[:,None,:]!=0, tf.int32)

        one_hot_b_mat = tf.tile(tf.one_hot(b_mat[:, None, :, :], depth=self.n_mb * self.n_mb + 1), (1, self.h, 1, 1, 1))
        W2_ = tf.einsum('BhdD,CBh->ChdD', self.W2, tf.nn.softmax(self.alpha2, 1))
        out = tf.einsum('bhij, bhijC, ChdD, bhjd -> bhiD', p_attn, one_hot_b_mat, W2_, value)
        return out

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        print(input_shape[1])
        return input_shape[1]

    def get_config(self):
        config = {'n_mb': self.n_mb, 'h': self.h, 'k': self.k}
        base_config = super(MbValueLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ChannelDense(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, channel, act=None, **kwargs):
        super(ChannelDense, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.channel = channel
        self.act = act

    def build(self, input_shape):
        super(ChannelDense, self).build(input_shape)
        self.wt = self.add_weight(shape=(self.channel, self.input_dim, self.output_dim), initializer=truncated_normal(stddev=0.02), dtype=tf.float32, name='ch_ds_wt')
        self.bias = self.add_weight(shape=(self.channel, self.output_dim), initializer=truncated_normal(stddev=0.02), dtype=tf.float32, name='ch_ds_bias')

    def call(self, inputs, mask=None):  # [b, n, C, d]
        t1 = tf.einsum('bnCd,Cdk->bnCk', inputs, self.wt)
        out = t1 + self.bias
        if self.act is not None:
            out = tf.keras.activations.gelu(out)
        return out

    def compute_output_shape(self, input_shape=None):
        return input_shape

    def get_config(self):
        config = {'input_dim': self.input_dim, 'output_dim': self.output_dim, 'channel': self.channel, 'act': self.act}
        base_config = super(ChannelDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MbTransformer(object):
    def __init__(self, seq_length, num_hidden_layers, num_attention_heads, size_per_head, n_mb=None, b_qkv=None, b_ff=None, b_head=None, b_value=None, b_pe=None,
                 n_moe=None, n_c=None, n_attr=None):
        self.seq_length = seq_length
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head
        self.input_width = self.num_attention_heads*self.size_per_head
        self.hidden_size = self.input_width
        self.intermediate_size = self.hidden_size*4

        # 模型内部参数
        self.initializer_range = 0.02
        self.hidden_dropout_prob = 0.1
        self.intermediate_act_fn = self.gelu
        self.query_act_fn = self.gelu
        self.key_act_fn = self.gelu
        self.value_act_fn = self.gelu
        self.hidden_act_fn = self.gelu

        self.n_mb = n_mb
        self.b_qkv = b_qkv
        self.b_ff = b_ff
        self.b_head = b_head
        self.b_value = b_value
        self.b_pe = b_pe
        self.n_moe = n_moe
        self.n_c = n_c
        self.n_attr = n_attr

    def attention_layer(self, from_tensor, to_tensor, b_seq=None, attention_mask=None, c_seq=None, da_scores=None):
        # from_tensor: [batch_size*seq_length, input_width]
        if self.b_qkv and self.n_mb > 1:       # 不同行为类型不同Q、K、V变换
            n_mb_1 = self.n_mb+1
            query_layer = Dense(self.hidden_size * n_mb_1, activation=self.query_act_fn,
                                kernel_initializer=self.create_initializer())(from_tensor)
            key_layer = Dense(self.hidden_size * n_mb_1, activation=self.key_act_fn,
                              kernel_initializer=self.create_initializer())(to_tensor)
            value_layer = Dense(self.hidden_size * n_mb_1, activation=self.value_act_fn,
                                kernel_initializer=self.create_initializer())(to_tensor)
            query_layer = Reshape([self.seq_length, self.hidden_size, n_mb_1])(query_layer)
            key_layer = Reshape([self.seq_length, self.hidden_size, n_mb_1])(key_layer)
            value_layer = Reshape([self.seq_length, self.hidden_size, n_mb_1])(value_layer)
            b_hot = Lambda(lambda x: tf.one_hot(x-1, n_mb_1))(b_seq)
            query_layer = Lambda(lambda x: tf.einsum('bndB,bnB->bnd', x[0], x[1]))([query_layer, b_hot])
            key_layer = Lambda(lambda x: tf.einsum('bndB,bnB->bnd', x[0], x[1]))([key_layer, b_hot])
            value_layer = Lambda(lambda x: tf.einsum('bndB,bnB->bnd', x[0], x[1]))([value_layer, b_hot])
        else:
            query_layer = Dense(self.hidden_size, activation=self.query_act_fn,
                                kernel_initializer=self.create_initializer())(from_tensor)
            key_layer = Dense(self.hidden_size, activation=self.key_act_fn,
                              kernel_initializer=self.create_initializer())(to_tensor)
            value_layer = Dense(self.hidden_size, activation=self.value_act_fn,
                                kernel_initializer=self.create_initializer())(to_tensor)

        # [batch_size, num_attention_heads, seq_length, size_per_head]
        query_layer = self.transpose_for_scores(query_layer, self.num_attention_heads, self.seq_length)
        key_layer = self.transpose_for_scores(key_layer, self.num_attention_heads, self.seq_length)
        if self.b_head and self.n_mb > 1:
            attention_scores = MbHeadLayer(n_mb=self.n_mb, h=self.num_attention_heads, k=self.size_per_head)([query_layer, key_layer, b_seq])
        else:
            attention_scores = Lambda(lambda x: tf.matmul(x[0], x[1], transpose_b=True))([query_layer, key_layer])
        if self.b_pe == "pos":   # 相对位置编码
            attention_scores = PeLayer(seq_len=self.seq_length, n_mb=self.n_mb, h=self.num_attention_heads, max_distance=self.seq_length)([attention_scores, b_seq])
        elif self.b_pe == 'da':
            if c_seq is None:
                n_c = None
            else:
                n_c = self.n_c
            if da_scores is None:
                attention_scores = DisAttLayer(seq_len=self.seq_length, n_mb=self.n_mb, h=self.num_attention_heads, n_c=n_c, n_attr=self.n_attr)(
                    [attention_scores, b_seq, c_seq])
            else:
                attention_scores = Add()([attention_scores, da_scores])

        if attention_mask is not None:
            # [batch_size, 1, seq_length]
            attention_mask = Lambda(lambda x: tf.expand_dims(x, axis=[1]))(attention_mask)
            adder = Lambda(lambda x: (1.0 - tf.cast(x, tf.float32)) * -10000.0)(attention_mask)
            attention_scores = Add()([attention_scores, adder])
        # [batch_size, num_attention_heads, seq_length, seq_length]
        attention_scores = DivConstant(math.sqrt(float(self.size_per_head)))(attention_scores)
        attention_prob = Softmax()(attention_scores)
        if attention_mask is not None:
            attention_prob = Lambda(lambda x: x[0]*tf.cast(x[1], dtype=tf.float32))([attention_prob, attention_mask])
        value_layer = Reshape([self.seq_length, self.num_attention_heads, -1])(value_layer)
        value_layer = Lambda(lambda x: tf.transpose(x, [0, 2, 1, 3]))(value_layer)
        # attention_prob: [batch_size, num_attention_heads, seq_length, seq_length]
        # value_layer: [batch_size, num_attention_heads, seq_length, size_per_head]
        if self.b_value and self.n_mb > 1:
            context_layer = MbValueLayer(n_mb=self.n_mb, h=self.num_attention_heads, k=self.size_per_head)([attention_prob, value_layer, b_seq])
        else:
            context_layer = Lambda(lambda x: tf.matmul(x[0], x[1]))([attention_prob, value_layer])
        context_layer = Lambda(lambda x: tf.transpose(x, [0, 2, 1, 3]))(context_layer)

        context_layer = Reshape([self.seq_length, self.hidden_size])(context_layer)
        return context_layer

    @staticmethod
    def gelu(x):
        cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
        return x * cdf

    @staticmethod
    def transpose_for_scores(input_tensor, num_attention_heads, seq_length):
        output_tensor = Reshape([seq_length, num_attention_heads, -1])(input_tensor)
        output_tensor = Lambda(lambda x: tf.transpose(x, [0, 2, 1, 3]))(output_tensor)
        return output_tensor

    @staticmethod
    def auc(y_true, y_prob):
        return tf.py_function(roc_auc_score, (y_true, y_prob), tf.double)

    def create_initializer(self):
        return truncated_normal(stddev=self.initializer_range)

    def transform(self, input_tensor, attention_mask, b_seq=None, get_first=False, c_seq=None, to_tensor=None):
        # input_tensor: [batch_size, seq_length, input_width] -> [batch_size, seq_length, input_width]
        prev_output = input_tensor
        all_layer_outputs = []
        da_scores = Sian(seq_len=self.seq_length, n_mb=self.n_mb, h=self.num_attention_heads, n_c=self.n_c, n_attr=self.n_attr)([b_seq, c_seq])

        for layer_idx in range(self.num_hidden_layers):
            with variable_scope("layer_%d" % layer_idx):
                layer_input = prev_output
                to_tensor_in = layer_input
                if layer_idx == 0 and to_tensor is not None:
                    to_tensor_in = to_tensor
                with variable_scope("attention"):
                    attention_heads = []
                    with variable_scope("self"):
                        attention_head = self.attention_layer(from_tensor=layer_input, to_tensor=to_tensor_in, attention_mask=attention_mask, b_seq=b_seq,
                                                              c_seq=c_seq, da_scores=da_scores)
                        attention_heads.append(attention_head)
                    assert len(attention_heads) == 1
                    attention_out = attention_heads[0]

                    with variable_scope("output"):
                        # attention_out: [batch_size * seq_length, input_width]
                        # attention_out = Dense(self.hidden_size, kernel_initializer=self.create_initializer())(attention_out)
                        attention_out = Dropout(self.hidden_dropout_prob)(attention_out)
                        attention_out = Add()([attention_out, layer_input])
                        attention_out = LayerNormalization()(attention_out)

                        if self.b_ff and self.n_mb > 1:
                            n_mb_1 = self.n_mb + 1
                            intermediate_output = Dense(self.intermediate_size*n_mb_1, activation=self.intermediate_act_fn,
                                                        kernel_initializer=self.create_initializer())(attention_out)
                            b_hot = Lambda(lambda x: tf.one_hot(x-1, n_mb_1))(b_seq)
                            intermediate_output = Reshape([self.seq_length, n_mb_1, self.intermediate_size])(intermediate_output)
                            intermediate_output = Dropout(self.hidden_dropout_prob)(intermediate_output)
                            # layer_output = Dense(self.hidden_size, kernel_initializer=self.create_initializer())(intermediate_output)
                            layer_output = ChannelDense(self.intermediate_size, self.hidden_size, n_mb_1)(intermediate_output)
                            layer_output = Lambda(lambda x: tf.einsum('bnBd,bnB->bnd', x[0], x[1]))([layer_output, b_hot])
                        elif self.n_moe:
                            n_mb_1 = self.n_moe
                            intermediate_output = Dense(self.intermediate_size * n_mb_1,
                                                        activation=self.intermediate_act_fn,
                                                        kernel_initializer=self.create_initializer())(attention_out)
                            gate = Dense(self.n_moe, activation='softmax', use_bias=False)(attention_out)
                            intermediate_output = Reshape([self.seq_length, n_mb_1, self.intermediate_size])(
                                intermediate_output)
                            intermediate_output = Dropout(self.hidden_dropout_prob)(intermediate_output)
                            layer_output = ChannelDense(self.intermediate_size, self.hidden_size, self.n_mb)(intermediate_output)
                            layer_output = Lambda(lambda x: tf.einsum('bnBd,bnB->bnd', x[0], x[1]))(
                                [layer_output, gate])
                        else:
                            intermediate_output = Dense(self.intermediate_size, activation=self.intermediate_act_fn,
                                                        kernel_initializer=self.create_initializer())(attention_out)
                            intermediate_output = Dropout(self.hidden_dropout_prob)(intermediate_output)
                            layer_output = Dense(self.hidden_size, kernel_initializer=self.create_initializer())(intermediate_output)
                        layer_output = Dropout(self.hidden_dropout_prob)(layer_output)
                        layer_output = Add()([layer_output, attention_out])
                        layer_output = LayerNormalization()(layer_output)
                        prev_output = layer_output
                        all_layer_outputs.append(layer_output)
        # [batch_size, seq_length, input_width]
        final_output = prev_output
        if get_first:
            return final_output[:, :1, :]
        else:
            return final_output

    @staticmethod
    def get_custom_objects():
        return {"gelu": MbTransformer.gelu, "BatchGather": BatchGather, "DivConstant": DivConstant}
