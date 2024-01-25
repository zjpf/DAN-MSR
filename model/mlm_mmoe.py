import sys

from dan_exp.bert.transformer import Transformer
import tensorflow as tf
from dan_exp.bert.bert_layer import PositionEmbedding, Dim2Mask, LossLayer, BatchGather, DivConstant, Dim2MaskFst, PrintLayer, NceLossLayer
from tensorflow.python.keras.utils.vis_utils import plot_model
from .bert4rec import MlmLossLayer, CausalLayer, PositionEmbedding

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
TruncatedNormal = keras.initializers.TruncatedNormal


# 实现MMOE输出
class MbMlmLossLayer(tf.keras.layers.Layer):
    def __init__(self, tied_to, hidden_size, voc_size, n_e_sh, n_e_sp, n_mb=4, **kwargs):
        super(MbMlmLossLayer, self).__init__(**kwargs)
        self.tied_to = tied_to
        self.voc_size = voc_size
        self.n_mb = n_mb
        self.hidden_size = hidden_size
        self.n_e_sh = n_e_sh
        self.n_e_sp = n_e_sp

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        super(MbMlmLossLayer, self).build(input_shape)
        self.weight_sh = []
        self.bias_sh = []
        for i in range(self.n_e_sh):
            w1 = self.add_weight(shape=(self.hidden_size, self.hidden_size), initializer=TruncatedNormal(stddev=0.02), dtype=tf.float32, name='weight_sh_{}'.format(i))
            self.weight_sh.append(w1)
            b1 = self.add_weight(shape=(self.hidden_size,), initializer=TruncatedNormal(stddev=0.02), dtype=tf.float32, name='bias_sh_{}'.format(i))
            self.bias_sh.append(b1)
        self.weight_sp = []
        self.bias_sp = []
        for i in range(self.n_mb*self.n_e_sp):
            w1 = self.add_weight(shape=(self.hidden_size, self.hidden_size), initializer=TruncatedNormal(stddev=0.02), dtype=tf.float32, name='weight_sp_{}'.format(i))
            self.weight_sp.append(w1)
            b1 = self.add_weight(shape=(self.hidden_size,), initializer=TruncatedNormal(stddev=0.02), dtype=tf.float32, name='bias_sp_{}'.format(i))
            self.bias_sp.append(b1)
        self.w_gates = self.add_weight(shape=(self.n_mb, self.hidden_size, self.n_e_sh+self.n_e_sp), initializer=TruncatedNormal(stddev=0.02), dtype=tf.float32, name='w_gates')
        self.bias = self.add_weight(shape=(self.voc_size,), initializer=keras.initializers.TruncatedNormal(stddev=0.02), dtype=tf.float32, name='bias')

    def call(self, inputs, mask=None):
        assert isinstance(inputs, list)
        x, labels, xb = inputs  # [b, n, d], [b, n], [b, n]
        x = tf.reshape(x, [-1, self.hidden_size])
        labels = tf.reshape(labels, [-1])
        xb = tf.reshape(xb, [-1])
        label_msk = tf.reshape(tf.where(labels > 0), [-1])
        labels = tf.gather(labels, label_msk)
        x = tf.gather(x, label_msk)
        xb = tf.gather(xb, label_msk)
        # 专家网络
        shared_eo = []
        for i in range(self.n_e_sh):
            eo = tf.matmul(x, self.weight_sh[i])
            eo = tf.nn.bias_add(eo, self.bias_sh[i])
            shared_eo.append(eo)
        specific_eo = []
        for i in range(self.n_mb*self.n_e_sp):
            eo = tf.matmul(x, self.weight_sp[i])
            eo = tf.nn.bias_add(eo, self.bias_sp[i])
            specific_eo.append(eo)
        gates_o = tf.nn.softmax(tf.einsum('nd,tde->tne', x, self.w_gates))
        mb_eo = []
        for i in range(self.n_mb):
            b_eo = tf.stack(shared_eo + specific_eo[i*self.n_e_sp:(i+1)*self.n_e_sp])
            mb_eo.append(b_eo)
        merge_eo = tf.stack(mb_eo)
        output = tf.einsum('tend,tne->tnd', merge_eo, gates_o)
        outputs = tf.concat([tf.expand_dims(tf.zeros_like(x), 0), output], axis=0)
        e_out = tf.einsum('tnd,nt->nd', outputs, tf.cast(tf.one_hot(xb, self.n_mb+1), tf.float32))
        # add & layer norm
        e_out = tf.reshape(e_out, (-1, self.hidden_size))
        e_out = LayerNormalization()(e_out) + x

        # 产生概率
        prob = tf.matmul(e_out, tf.transpose(self.tied_to.weights[0]))
        prob = tf.nn.bias_add(prob, self.bias)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=prob)
        self.add_loss(loss)
        prob = tf.nn.softmax(prob)
        return prob

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        print(input_shape[0][0], self.voc_size)
        return input_shape[0][0], self.voc_size

    def get_config(self):
        config = {'tied_to': self.tied_to, 'voc_size': self.voc_size, 'hidden_size': self.hidden_size, 'n_e_sh': self.n_e_sh,
                  'n_e_sp': self.n_e_sp, 'n_mb': self.n_mb}
        base_config = super(MbMlmLossLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MlmMmoe(object):
    def __init__(self, seq_length, voc_size, num_hidden_layers, num_attention_heads, size_per_head, n_e_sh, n_e_sp, n_mb, n_moe=None, b_event=None,
                 label_bhv='behaviors', b_qkv=None, b_ff=None, b_head=None, b_value=None, b_pe=None, causal=False, b_dt=None,
                 n_gate=None, b_cate=None, n_c=None, n_attr=None):
        hidden_size = num_attention_heads*size_per_head
        if b_qkv or b_ff or b_head or b_value or b_pe or b_event.startswith('fusion'):
            from dan_exp.bert.moe_transformer import MbTransformer
            transformer = MbTransformer(seq_length=seq_length, num_hidden_layers=num_hidden_layers, num_attention_heads=num_attention_heads, size_per_head=size_per_head,
                                        n_mb=n_mb, b_qkv=b_qkv, b_ff=b_ff, b_head=b_head, b_value=b_value, b_pe=b_pe, n_moe=n_moe, n_c=n_c, n_attr=n_attr)
        elif b_event in ['nova', 'dif_sr']:
            from dan_exp.bert.dif_sr import DifTransformer
            transformer = DifTransformer(seq_length=seq_length, num_hidden_layers=num_hidden_layers,
                                         num_attention_heads=num_attention_heads, size_per_head=size_per_head, n_mb=n_mb, mode=b_event)
        else:
            raise ValueError("No")
        embedding = Embedding(input_dim=voc_size, output_dim=hidden_size, embeddings_initializer=transformer.create_initializer())

        input_list = []
        f_items = Input(shape=(seq_length, ), dtype='int32', name="items")
        f_labels = Input(shape=(seq_length, ), dtype='int32', name="labels")
        f_seq_length = Input(shape=(1,), dtype='int32', name='seq_length')
        if label_bhv == "behaviors":
            f_bhv = Input(shape=(seq_length,), dtype='int32', name="behaviors")
            f_label_bhv = f_bhv
            input_list.extend([f_items, f_labels, f_seq_length, f_bhv])
        else:
            raise ValueError("No")
        input_mask = Lambda(lambda x: tf.sequence_mask(tf.squeeze(x), maxlen=seq_length))(f_seq_length)
        attention_mask = Dim2Mask(seq_length)(input_mask)
        if causal:      # 1. 下三角矩阵实现因果attention
            attention_mask = Lambda(lambda x: tf.linalg.band_part(x, -1, 0))(attention_mask)

        event_embedding = Embedding(input_dim=n_mb + 1, output_dim=hidden_size,
                                    embeddings_initializer=transformer.create_initializer())
        event_emb = event_embedding(f_bhv)
        raw_item_emb = embedding(f_items)
        if b_event == 'fusion-add':  # 属性融合
            item_emb = Add()([raw_item_emb, event_emb])
            item_emb = PositionEmbedding(seq_length, hidden_size)(item_emb)
        # elif b_event == 'fusion-concat':  # 属性融合
        #     item_emb = Concatenate()([raw_item_emb, event_emb])
        #     fusion_emb = PositionEmbedding(seq_length, hidden_size, merge_mode='concat')(item_emb)
        #     item_emb = Dense(hidden_size, activation='relu', kernel_initializer=transformer.create_initializer())(fusion_emb)
        # elif b_event == 'fusion-gate':  # 属性融合
        #     from dan_exp.bert.dif_sr import GateFusionLayer
        #     item_emb = GateFusionLayer(hidden_size=hidden_size, n_gate=3, seq_len=seq_length)([raw_item_emb, event_emb])
        elif b_event in ['nova', 'dif_sr']:
            pos_emb = PositionEmbedding(seq_length, hidden_size, merge_mode='non')(raw_item_emb)
            attr_emb_s = [pos_emb, event_emb]
            item_emb = raw_item_emb
        else:
            item_emb = raw_item_emb
        if b_pe == 'raw':  # 2. 原始的位置编码position embedding
            item_emb = PositionEmbedding(seq_length, hidden_size)(item_emb)

        if b_cate and n_attr is not None:
            f_cate = Input(shape=(seq_length, n_attr+1,), dtype='float32', name="cate")
            input_list.append(f_cate)
            cate_emb = Dense(hidden_size, kernel_initializer=transformer.create_initializer())(f_cate)
            if (b_qkv and b_ff and b_head and b_value and b_pe) or b_event in ['fusion-add']:
                item_emb = Add()([item_emb, cate_emb])
            elif b_event in ['nova', 'dif_sr']:
                attr_emb_s.append(cate_emb)
        elif b_cate:  # idea4：cate_id特征输入
            f_cate = Input(shape=(seq_length,), dtype='int32', name="cate")
            input_list.append(f_cate)
            cate_embedding = Embedding(input_dim=n_c + 2, output_dim=hidden_size, embeddings_initializer=transformer.create_initializer())
            cate_emb = cate_embedding(f_cate)
            if (b_qkv and b_ff and b_head and b_value and b_pe) or b_event in ['fusion-add']:
                item_emb = Add()([item_emb, cate_emb])
            elif b_event in ['nova', 'dif_sr']:
                attr_emb_s.append(cate_emb)
        else:
            f_cate = None

        if b_qkv or b_ff or b_head or b_value or b_pe in ['pos', 'da', 'raw'] or b_event.startswith('fusion'):  # 多行为PE、dan
            final_output = transformer.transform(item_emb, attention_mask, b_seq=f_bhv, c_seq=f_cate)
        elif b_event in ['dif_sr']:
            final_output = transformer.transform(item_emb, attention_mask, attr_emb_s=attr_emb_s)
        elif b_event in ['nova']:
            final_output = transformer.transform(item_emb, attention_mask, attr_emb_s=attr_emb_s)
        else:
            raise ValueError("No")
        if (n_e_sh is None or n_e_sp is None) or n_e_sh + n_e_sp == 0:  # 可选mlm、mmoe-mlm输出预测
            predict_out = MlmLossLayer(tied_to=embedding, voc_size=voc_size, hidden_size=hidden_size)([final_output, f_labels])
        else:
            predict_out = MbMlmLossLayer(tied_to=embedding, voc_size=voc_size, hidden_size=hidden_size, n_e_sh=n_e_sh, n_e_sp=n_e_sp, n_mb=n_mb)([final_output, f_labels, f_label_bhv])
        self.model = Model(inputs=input_list, outputs=predict_out)

    @staticmethod
    def get_custom_objects():
        return {"gelu": Transformer.gelu, "LossLayer": LossLayer, "BatchGather": BatchGather,
                "Dim2Mask": Dim2Mask, "DivConstant": DivConstant, "PositionEmbedding": PositionEmbedding, "Dim2MaskFst": Dim2MaskFst}


if __name__ == "__main__":
    # 1. 模型训练
    print(tf.executing_eagerly())
    # model = MlmMmoe(seq_length=200, voc_size=2000, num_hidden_layers=2, num_attention_heads=2, size_per_head=4, n_e_sh=3, n_e_sp=2, n_mb=4).model
    model = MlmMmoe(seq_length=200, voc_size=2000, num_hidden_layers=2, num_attention_heads=2, size_per_head=4,
                    n_e_sh=3, n_e_sp=2, n_mb=4, b_qkv=True, b_ff=True, b_head=True, b_value=True, b_pe=True).model
    plot_model(model, to_file="../data/mlm_mmoe.png", show_shapes=False)
    print("END")
