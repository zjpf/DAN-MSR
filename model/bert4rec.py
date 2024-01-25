import sys

from dan_exp.bert.transformer import Transformer
import tensorflow as tf
from dan_exp.bert.bert_layer import PositionEmbedding, Dim2Mask, LossLayer, BatchGather, DivConstant, Dim2MaskFst, PrintLayer, NceLossLayer
from tensorflow.python.keras.utils.vis_utils import plot_model

keras = tf.keras
Embedding = keras.layers.Embedding
Input = keras.Input
Reshape = keras.layers.Reshape
Concatenate = keras.layers.Concatenate
BatchNormalization = keras.layers.BatchNormalization
Add = keras.layers.Add
Dropout = keras.layers.Dropout
Dense = keras.layers.Dense
LayerNormalization = keras.layers.LayerNormalization
Model = keras.models.Model
Lambda = tf.keras.layers.Lambda


class MlmLossLayer(tf.keras.layers.Layer):
    def __init__(self, tied_to, hidden_size, voc_size, **kwargs):
        super(MlmLossLayer, self).__init__(**kwargs)
        self.tied_to = tied_to
        self.voc_size = voc_size
        self.hidden_size = hidden_size

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        super(MlmLossLayer, self).build(input_shape)
        self.bias = self.add_weight(shape=(self.voc_size,), initializer=keras.initializers.TruncatedNormal(stddev=0.02), dtype=tf.float32)

    def call(self, inputs, mask=None):
        assert isinstance(inputs, list)
        transform_out, labels = inputs  # [b, n, d], [b, n]

        transform_out = tf.reshape(transform_out, [-1, self.hidden_size])
        labels = tf.reshape(labels, [-1])
        label_msk = tf.reshape(tf.where(labels > 0), [-1])
        # label_msk = tf.experimental.numpy.nonzero(labels)[0]
        labels = tf.gather(labels, label_msk)

        transform_msk = tf.gather(transform_out, label_msk)
        prob = tf.matmul(transform_msk, tf.transpose(self.tied_to.weights[0]))
        prob = tf.nn.bias_add(prob, self.bias)

        raw_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=prob)
        loss = tf.reduce_sum(raw_loss)
        self.add_loss(raw_loss)
        # self.add_metric(raw_loss, name='raw_loss')
        # self.add_metric(loss, name='raw_sum_loss')
        prob = tf.nn.softmax(prob)
        return prob

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        print(input_shape[0][0], self.voc_size)
        return input_shape[0][0], self.voc_size

    def get_config(self):
        config = {'tied_to': self.tied_to, 'voc_size': self.voc_size, 'hidden_size': self.hidden_size}
        base_config = super(MlmLossLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CausalLayer(tf.keras.layers.Layer):
    def __init__(self, seq_length, **kwargs):
        super(CausalLayer, self).__init__(**kwargs)
        self.seq_length = seq_length

    def build(self, input_shape):
        super(CausalLayer, self).build(input_shape)

    def causal(self, ind):
        seq_length = self.seq_length
        b_msk_s = []
        msk_s = []
        prior_bi = 0
        prior_i = -1
        for bi in range(ind.shape[0]):
            if ind[bi][0] != prior_bi and prior_i != -1:
                msk_s.append(tf.fill((seq_length - prior_i-1, seq_length), 0.))
                b_msk_s.append(tf.concat(msk_s, axis=0))
                msk_s = []
                prior_i = -1
            a1 = tf.fill((ind[bi][1] - prior_i, ind[bi][1]+1), 1.)
            a2 = tf.fill((ind[bi][1] - prior_i, seq_length - ind[bi][1]-1), 0.)
            msk_s.append(tf.concat([a1, a2], -1))
            prior_i = ind[bi][1]
            prior_bi = ind[bi][0]
        msk_s.append(tf.fill((seq_length - prior_i-1, seq_length), 0.))
        b_msk_s.append(tf.concat(msk_s, axis=0))
        out = tf.stack(b_msk_s, axis=0)
        return out

    def call(self, inputs, mask=None):
        f_labels, attention_mask = inputs  # [b, n]
        # ind = tf.where(f_labels > 0)
        # out = tf.py_function(self.causal, [ind], Tout=tf.float32)
        # # tf.print(f_labels, summarize=300)
        # # tf.print(out, summarize=300)
        # out.set_shape(f_labels.shape+[self.seq_length])
        msk = 1.0-tf.one_hot(tf.range(1, self.seq_length+1), self.seq_length)
        out = msk*attention_mask
        return out

    def compute_output_shape(self, input_shape):
        return input_shape[0]+[self.seq_length]

    def get_config(self):
        config = {'seq_length': self.seq_length}
        base_config = super(CausalLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Bert4Rec(object):
    def __init__(self, seq_length, voc_size, num_hidden_layers, num_attention_heads, size_per_head, causal=False):
        hidden_size = num_attention_heads*size_per_head
        transformer = Transformer(seq_length=seq_length, num_hidden_layers=num_hidden_layers,
                                  num_attention_heads=num_attention_heads, size_per_head=size_per_head)
        embedding = Embedding(input_dim=voc_size, output_dim=hidden_size, embeddings_initializer=transformer.create_initializer())

        input_list = []
        f_items = Input(shape=(seq_length, ), dtype='int32', name="items")
        f_labels = Input(shape=(seq_length, ), dtype='int32', name="labels")
        f_seq_length = Input(shape=(1,), dtype='int32', name='seq_length')
        input_list.extend([f_items, f_labels, f_seq_length])

        input_mask = Lambda(lambda x: tf.sequence_mask(tf.squeeze(x), maxlen=seq_length))(f_seq_length)
        attention_mask = Dim2Mask(seq_length)(input_mask)
        if causal:
            # attention_mask = Lambda(lambda x: tf.linalg.band_part(x, -1, 0))(attention_mask)
            attention_mask = CausalLayer(seq_length=seq_length)([f_labels, attention_mask])

        item_emb = embedding(f_items)
        final_output = transformer.transform(item_emb, attention_mask)
        predict_out = MlmLossLayer(tied_to=embedding, voc_size=voc_size, hidden_size=hidden_size)([final_output, f_labels])
        self.model = Model(inputs=input_list, outputs=predict_out)

    @staticmethod
    def get_custom_objects():
        return {"gelu": Transformer.gelu, "LossLayer": LossLayer, "BatchGather": BatchGather,
                "Dim2Mask": Dim2Mask, "DivConstant": DivConstant, "PositionEmbedding": PositionEmbedding, "Dim2MaskFst": Dim2MaskFst}


if __name__ == "__main__":
    # 1. 模型训练
    model = Bert4Rec(seq_length=200, voc_size=2000, num_hidden_layers=2, num_attention_heads=2, size_per_head=4).model
    plot_model(model, to_file="../bert4rec.png", show_shapes=False)
    print("END")
