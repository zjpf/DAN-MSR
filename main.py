import math
import sys
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
from bert.bert_layer import ModelSaver
import yaml
import functools


def auc(y_true, y_prob):
    return tf.py_function(roc_auc_score_fixed, (y_true, y_prob), tf.double)


def roc_auc_score_fixed(y_true, y_prob):
    if len(np.unique(y_true)) == 1:
        print("Warn: only one class in y_true")
        return accuracy_score(y_true, np.rint(y_prob))
    return roc_auc_score(y_true, y_prob)


def train(conf, skip=False):
    model_name = conf['train']['model_name']
    seq_len = conf['data']['seq_len']
    buffer_size = conf['data'].get('buffer_size', 2000)
    data_type = conf['data']['data_type']
    batch_size = conf['train']['batch_size']
    num_labels = conf['data']['num_items']+2

    from model import get_model
    model = get_model(conf)
    from dataset import get_dataset
    gs, output_signature = get_dataset(conf)

    gen = functools.partial(gs.gen_features, 'train')
    x_y = tf.data.Dataset.from_generator(gen, output_signature=output_signature)\
        .shuffle(buffer_size=buffer_size).batch(batch_size)
    gen_eval = functools.partial(gs.gen_features, 'eval')
    x_y_eval = tf.data.Dataset.from_generator(gen_eval, output_signature=output_signature) \
        .shuffle(buffer_size=buffer_size).batch(batch_size)

    model_path = "../data/saved_model/{}_{}_{}_model_{}_{}.h5".format(conf['data']['name'], model_name, data_type, conf['train']['exp'], seq_len)
    callbacks = [EarlyStopping(monitor='val_hit', patience=conf['train']['patience'], mode="max"),
                 ReduceLROnPlateau(monitor='val_hit', factor=0.3, patience=1, min_lr=1e-8, mode='max'),
                 ModelSaver("val_hit", model, model_path=model_path, mode='max')]  # val_loss
    model.compile("adam", loss=None, metrics=hr_top(num_labels, batch_size))
    if model_name in ["mlmMmoe", "mbStr", "bert4rec"]:
        model.run_eagerly = True
    if not skip:
        model.fit(x_y, epochs=conf['train']['epochs'], verbose=1, validation_data=x_y_eval, callbacks=callbacks)
    model.load_weights(model_path)
    return model, x_y_eval


def predict(model, gen, num_items, score_fn, md='top10'):
    import random
    rn_dis = {}
    score_fn = '../data/score_oot'+score_fn
    with open(score_fn, 'w') as f:
        if md == 'top10':
            for x, y in gen:
                y_p = model.predict(x)
                for bi in range(len(y_p)):
                    out = []
                    for n in range(100):
                        top = np.argmax(y_p[bi], axis=-1)
                        out.append("{}#{}".format(top, y_p[bi][top]))
                        y_p[bi][top] = -sys.maxsize
                    if y.dtype == tf.string:
                        f.write("{},{}\n".format(y[bi].numpy().decode('utf-8'), "|".join(out)))
                    else:
                        tgt = [str(t) for t in y[bi].numpy() if t >= 0]
                        f.write("{},{}\n".format('^'.join(tgt), "|".join(out)))
        elif md == "rand":
            for x, y in gen:
                y_p = model.predict(x)
                for bi in range(len(y_p)):
                    out = []
                    rd_n = set(random.sample(range(1, num_items), 99))
                    rd_n.add(int(y[bi]))
                    while len(rd_n) < 100:
                        rep = random.sample(range(1, num_items), 100-len(rd_n))
                        rd_n.update(rep)
                    for n in rd_n:
                        out.append((n, y_p[bi][n]))
                    sorted_out = sorted(out, key=lambda t: t[1], reverse=True)
                    out = []
                    for tp in sorted_out:
                        out.append("{}#{}".format(tp[0], tp[1]))
                    f.write("{},{}\n".format(y[bi], "|".join(out)))
    for tp in sorted(rn_dis.items(), key=lambda k: k[0]):
        print(tp)
    return score_fn


def hit_ratio(score_fn, n=10):
    import csv
    import math
    user_num = 0
    hit_num = 0
    n_dcg = 0.0
    with open(score_fn, 'r') as f:
        for row in csv.reader(f):
            dcg, i_dcg = 0.0, 0.0
            label = set(row[0].split('^'))
            top_10 = set()
            rec_tps = row[1].split("|")
            for i in range(n):
                e = rec_tps[i].split("#")[0]
                top_10.add(e)
                if e in label:
                    dcg += 1.0/math.log2(i+2)
            i_dcg = 1. + sum([1.0/math.log2(i+3) for i in range(len(label)-1)])
            user_num += 1
            hit_num += len(label & top_10)*1.0/len(label)
            n_dcg += dcg/i_dcg
    print(score_fn, n, "hit_ratio: ", user_num, hit_num, hit_num/user_num, "n_dcg: ", n_dcg/user_num)


def hr_top(num_labels=10, batch_size=128, k=10):
    def hit(y_true, y_pred):  # [b, n], [b, M]
        # 计算top10 HR和n-dcg
        # tf.print(y_true)
        # tf.print(y_pred)
        # print(y_true, y_pred)
        y_pred = y_pred[:y_true.shape[0]]
        top_v, top_i = tf.math.top_k(y_pred, k=k)
        y_true = tf.cast(y_true, tf.int32)
        labels = tf.reduce_sum(tf.one_hot(y_true, num_labels), axis=-2)
        hit_labels = tf.gather(labels, top_i, batch_dims=1)
        hr = tf.reduce_sum(hit_labels, axis=-1) / tf.reduce_sum(tf.cast(y_true >= 0, tf.float32), axis=-1)
        return hr
    return hit


def main():
    import copy
    conf = {}
    for fn in sys.argv[1].split(','):
        with open(fn, encoding='utf-8') as fr:
            fn_conf = yaml.load(fr.read(), Loader=yaml.Loader)
            for k, vd in fn_conf.items():
                ov = conf.setdefault(k, {})
                ov.update(fn_conf[k])
            print(conf)
    with open('config/share.yaml', encoding='utf-8') as fr:
        sh_conf = yaml.load(fr.read(), Loader=yaml.Loader)
        for k, vd in sh_conf.items():
            conf[k].update(sh_conf[k])
        print(conf)
    mds = ['top10']
    skip = conf['train']['skip']
    raw_conf = conf
    with open(sys.argv[2], encoding='utf-8') as fr:  # "config/cmp-debug.yaml"
        fn_conf = yaml.load(fr.read(), Loader=yaml.Loader)
        for exp, exp_conf in fn_conf.items():
            conf = copy.deepcopy(raw_conf)
            for k, vd in exp_conf.items():
                conf[k].update(vd)
            print(exp, conf)
            # for debug
            conf['data']['buffer_size'] = 400
            conf['data']['seq_len'] = 32
            # continue
            m1, gen = train(conf=conf, skip=skip)
            for md in mds:
                score_fn = predict(model=m1, gen=gen, num_items=conf['data']['num_items'], md=md,
                                   score_fn="_{}_{}_{}_{}_{}_{}".format(conf['data']['name'], conf['data']['data_type'],
                                                                        conf['train']['model_name'], conf['train']['exp'], md, conf['data']['seq_len']))
                for n in [10, 20, 50, 100]:
                    hit_ratio(score_fn, n=n)


if __name__ == "__main__":
    import os
    if len(sys.argv) >= 3:
        gpu = sys.argv[-1]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    tf.executing_eagerly = True
    print(sys.path)
    main()
