from .mlm_cate_sample_v2 import MlmSample
from .sas_rec_sample import SasRecSample
import tensorflow as tf


def get_dataset(conf):
    data_type = conf['data'].get('data_type')
    n_attr = conf['data'].get('n_attr')
    seq_len = conf['data'].get('seq_len')
    if data_type == "mlm_cate" and n_attr is None:
        gs = MlmSample(conf)
        fs = {"items": tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),
              "labels": tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),
              "behaviors": tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),
              "seq_length": tf.TensorSpec(shape=(), dtype=tf.int32),
              "event_dt": tf.TensorSpec(shape=(seq_len,), dtype=tf.float32),
              "cate": tf.TensorSpec(shape=(seq_len,), dtype=tf.float32)}
        output_signature = (fs, tf.TensorSpec(shape=(), dtype=tf.string))
        output_signature = (fs, tf.TensorSpec(shape=(seq_len,), dtype=tf.int32))
    elif data_type == "mlm_cate" and n_attr > 0:
        gs = MlmSample(conf)
        fs = {"items": tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),
              "labels": tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),
              "behaviors": tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),
              "seq_length": tf.TensorSpec(shape=(), dtype=tf.int32),
              "event_dt": tf.TensorSpec(shape=(seq_len,), dtype=tf.float32),
              "cate": tf.TensorSpec(shape=(seq_len, n_attr+1,), dtype=tf.float32)}
        output_signature = (fs, tf.TensorSpec(shape=(), dtype=tf.string))
        output_signature = (fs, tf.TensorSpec(shape=(seq_len,), dtype=tf.int32))
    elif data_type == 'sas' and n_attr is None:
        gs = SasRecSample(conf)
        output_signature = ({"items": tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),
                             "labels": tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),
                             "behaviors": tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),
                             "ar_dt": tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),
                             "seq_length": tf.TensorSpec(shape=(), dtype=tf.int32),
                             "event_dt": tf.TensorSpec(shape=(seq_len,), dtype=tf.float32),
                             "cate": tf.TensorSpec(shape=(seq_len,), dtype=tf.float32)},
                            tf.TensorSpec(shape=(seq_len,), dtype=tf.int32))
    elif data_type == 'sas' and n_attr > 0:
        gs = SasRecSample(conf)
        output_signature = ({"items": tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),
                             "labels": tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),
                             "behaviors": tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),
                             "ar_dt": tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),
                             "seq_length": tf.TensorSpec(shape=(), dtype=tf.int32),
                             "event_dt": tf.TensorSpec(shape=(seq_len,), dtype=tf.float32),
                             "cate": tf.TensorSpec(shape=(seq_len, n_attr+1,), dtype=tf.float32)},
                            tf.TensorSpec(shape=(seq_len,), dtype=tf.int32))

    return gs, output_signature
