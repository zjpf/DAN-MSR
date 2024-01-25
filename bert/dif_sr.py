from .bert_layer import BatchGather, DivConstant
import tensorflow as tf
import numpy as np
import math
from .moe_transformer import PeLayer, MbHeadLayer, MbValueLayer, ChannelDense, DisAttLayer

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
TruncatedNormal = keras.initializers.TruncatedNormal


class GateFusionLayer(tf.keras.layers.Layer):
    """输入一个tensor list,输出gating加权后的tensor"""
    def __init__(self, hidden_size, n_gate, seq_len, **kwargs):
        super(GateFusionLayer, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.n_gate = n_gate
        self.seq_len = seq_len

    def build(self, input_shape):
        super(GateFusionLayer, self).build(input_shape)
        self.wt = self.add_weight(name='gate_wt', shape=(self.hidden_size, self.n_gate), initializer=TruncatedNormal(stddev=0.02))
        self.pe = self.add_weight(name='gate_pe', shape=(1, self.seq_len, self.hidden_size), initializer=TruncatedNormal(stddev=0.02))

    def call(self, inputs):  # 输入: [b,n,d]*m
        f_item, f_event = inputs
        input_shape = tf.shape(f_item)
        batch_size, seq_len = input_shape[0], input_shape[1]
        f_pos = tf.tile(self.pe, [batch_size, 1, 1])
        fs = tf.stack([f_item, f_event, f_pos])
        gate = tf.multiply(fs, self.wt)  # [b,n,m]
        o_gate = tf.einsum('mbnd,bnm->bnd', fs, gate)
        return o_gate

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = {'hidden_size': self.hidden_size, 'n_gate': self.n_gate}
        base_config = super(GateFusionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DifTransformer(object):
    def __init__(self, seq_length, num_hidden_layers, num_attention_heads, size_per_head, n_mb=None, b_qkv=None, b_ff=None, b_head=None, b_value=None, b_pe=None,
                 b_gate=None, n_moe=None, n_c=None, mode=None):
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
        self.b_gate = b_gate
        self.n_moe = n_moe
        self.n_c = n_c
        self.fusion_type = 'sum'
        self.mode = mode

    def attention_layer(self, from_tensor, to_tensor, attention_mask=None, attr_emb_s=None):
        # from_tensor: [batch_size*seq_length, input_width]
        # if True:
        #     query_layer = Dense(self.hidden_size, activation=self.query_act_fn,
        #                         kernel_initializer=self.create_initializer())(from_tensor)
        #     key_layer = Dense(self.hidden_size, activation=self.key_act_fn,
        #                       kernel_initializer=self.create_initializer())(to_tensor)
        #     value_layer = Dense(self.hidden_size, activation=self.value_act_fn,
        #                         kernel_initializer=self.create_initializer())(to_tensor)
        item_value_layer = Dense(self.hidden_size, activation=self.value_act_fn,
                                 kernel_initializer=self.create_initializer())(to_tensor)
        if self.mode == 'dif_sr':
            attr_scores = []
            for attr_emb in attr_emb_s+[from_tensor]:
                query_layer = Dense(self.hidden_size, activation=self.query_act_fn,
                                    kernel_initializer=self.create_initializer())(attr_emb)
                key_layer = Dense(self.hidden_size, activation=self.key_act_fn,
                                  kernel_initializer=self.create_initializer())(attr_emb)
                query_layer = self.transpose_for_scores(query_layer, self.num_attention_heads, self.seq_length)
                key_layer = self.transpose_for_scores(key_layer, self.num_attention_heads, self.seq_length)
                attr_scores.append(Lambda(lambda x: tf.matmul(x[0], x[1], transpose_b=True))([query_layer, key_layer]))
            if self.fusion_type == 'sum':
                attention_scores = Add()(attr_scores)
                attention_scores = DivConstant(math.sqrt(float(self.size_per_head)))(attention_scores)
        elif self.mode == 'nova':
            fusion_item = Add()(attr_emb_s+[from_tensor])
            query_layer = Dense(self.hidden_size, activation=self.query_act_fn,
                                kernel_initializer=self.create_initializer())(fusion_item)
            key_layer = Dense(self.hidden_size, activation=self.key_act_fn,
                              kernel_initializer=self.create_initializer())(fusion_item)
            query_layer = self.transpose_for_scores(query_layer, self.num_attention_heads, self.seq_length)
            key_layer = self.transpose_for_scores(key_layer, self.num_attention_heads, self.seq_length)
            attr_scores = Lambda(lambda x: tf.matmul(x[0], x[1], transpose_b=True))([query_layer, key_layer])
            attention_scores = DivConstant(math.sqrt(float(self.size_per_head)))(attr_scores)

        if attention_mask is not None:
            # [batch_size, 1, seq_length]
            attention_mask = Lambda(lambda x: tf.expand_dims(x, axis=[1]))(attention_mask)
            adder = Lambda(lambda x: (1.0 - tf.cast(x, tf.float32)) * -10000.0)(attention_mask)
            attention_scores = Add()([attention_scores, adder])
        # [batch_size, num_attention_heads, seq_length, seq_length]
        attention_prob = Softmax()(attention_scores)
        if attention_mask is not None:
            attention_prob = Lambda(lambda x: x[0]*tf.cast(x[1], dtype=tf.float32))([attention_prob, attention_mask])
        value_layer = Reshape([self.seq_length, self.num_attention_heads, -1])(item_value_layer)
        value_layer = Lambda(lambda x: tf.transpose(x, [0, 2, 1, 3]))(value_layer)
        # attention_prob: [batch_size, num_attention_heads, seq_length, seq_length]
        # value_layer: [batch_size, num_attention_heads, seq_length, size_per_head]
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

    def create_initializer(self):
        return truncated_normal(stddev=self.initializer_range)

    def transform(self, input_tensor, attention_mask, get_first=False, attr_emb_s=None):
        # input_tensor: [batch_size, seq_length, input_width] -> [batch_size, seq_length, input_width]
        prev_output = input_tensor
        all_layer_outputs = []

        for layer_idx in range(self.num_hidden_layers):
            with variable_scope("layer_%d" % layer_idx):
                layer_input = prev_output
                with variable_scope("attention"):
                    attention_heads = []
                    with variable_scope("self"):
                        attention_head = self.attention_layer(from_tensor=layer_input, to_tensor=layer_input, attention_mask=attention_mask, attr_emb_s=attr_emb_s)
                        attention_heads.append(attention_head)
                    assert len(attention_heads) == 1
                    attention_out = attention_heads[0]

                    with variable_scope("output"):  # attention_out: [batch_size * seq_length, input_width]
                        # attention_out = Dense(self.hidden_size, kernel_initializer=self.create_initializer())(attention_out)
                        attention_out = Dropout(self.hidden_dropout_prob)(attention_out)
                        attention_out = Add()([attention_out, layer_input])
                        attention_out = LayerNormalization()(attention_out)
                        if True:
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
        return {"gelu": DifTransformer.gelu, "BatchGather": BatchGather, "DivConstant": DivConstant}
