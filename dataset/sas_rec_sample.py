import csv
import tensorflow as tf
import random
import numpy as np
import datetime
import yaml
import sys
csv.field_size_limit(1024*1024*1024)


class SasRecSample:
    def __init__(self, conf):
        self.conf = conf
        self.train_start_dt = datetime.datetime.strptime(conf['data']['train_start_dt'], '%Y%m%d').date()    # yyyyMMdd格式整数
        self.train_end_dt = datetime.datetime.strptime(conf['data']['train_end_dt'], '%Y%m%d').date()
        self.eval_start_dt = datetime.datetime.strptime(conf['data']['eval_start_dt'], '%Y%m%d').date()
        self.eval_end_dt = datetime.datetime.strptime(conf['data']['eval_end_dt'], '%Y%m%d').date()
        self.standard_file = conf['data']['file_name']
        self.target_b = conf['data']['target_b']
        self.mb = eval(conf['data']['mb'])
        self.num_items = conf['data']['num_items']
        self.seq_len = conf['data']['seq_len']
        self.ar_prob = conf['data']['mask_prob']
        self.ar_bi = eval(conf['data']['mask_bi'])
        self.debug = conf['data']['debug']
        self.x = []
        self.y = []
        self.item_index = {}
        self.n_attr = conf['data'].get('n_attr', None)

    def gen_features(self, mode='train'):
        with open(self.standard_file) as f:
            for row in csv.reader(f):
                item_list, b_list, ts_list, c_list = [], [], [], []
                tps = row[1].split('|')
                # 1. 定位co_buy数
                i, co_buy, tgt_items, prior_ts, flag_train = -1, 0, set(), -1, 0
                while i + len(tps) >= 0:
                    item, event, cate, ts = tps[i].split('#')
                    if prior_ts == ts or prior_ts == -1 or self.standard_file.find('ijcai') > 0:
                        if prior_ts == -1:
                            prior_ts = ts
                        elif ts != prior_ts:
                            break
                        co_buy += 1
                        if event == self.target_b:
                            tgt_items.add(item)
                    else:
                        break
                    i -= 1
                if len(tps) - co_buy <= 0:
                    continue
                # if co_buy > 0:
                #     co_buy = co_buy - 1
                # elif co_buy == len(tps):
                #     co_buy = 0
                #     flag_train = 1
                for i in range(len(tps)-co_buy):
                    item, event, cate, ts = tps[i].split('#')
                    if len(ts) == 10:
                        dt = datetime.datetime.fromtimestamp(int(ts)).date()
                    else:
                        dt = self.train_start_dt
                    if self.debug or len(self.item_index) == 0:
                        item_list.append(int(item))
                    else:
                        item_list.append(self.item_index.get(item))
                    b_list.append(self.mb.get(event))
                    ts_list.append((dt-self.train_start_dt).days+1)
                    if self.n_attr is not None:
                        attr_c = [0] * (self.n_attr + 1)
                        for attr in cate.split('^'):
                            if len(attr) > 0:
                                attr_c[int(attr)] = 1
                        c_list.append(attr_c)
                    else:
                        c_list.append(int(cate))

                if mode == 'train':     # ToDO：ar_dt特征逻辑暂不考虑
                    p_dt = 0
                    ar_dt = [0]
                elif mode == 'eval':
                    p_dt = (self.eval_start_dt-self.train_start_dt).days+1
                    ar_dt = [p_dt]
                for i in range(2, len(ts_list)+1):
                    if ts_list[-i] == ts_list[-i + 1]:
                        ar_dt.append(p_dt)
                    else:
                        p_dt = ts_list[-i + 1]
                        ar_dt.append(p_dt)
                ar_dt.reverse()
                # if (b_list[-1] == self.mb[self.target_b] and len(item_list) <= 1) or (
                #         b_list[-1] != self.mb[self.target_b] and len(item_list) <= 1):  # len(item_list) < 2:
                #     continue
                if mode == 'train':
                    ar_items = []
                    for i in range(len(item_list)):
                        next_item = 0       # item索引从1开始不改为-1，下一次回归行为为label
                        if b_list[i] in self.ar_bi:
                            for j in range(i+1, len(item_list)):
                                if b_list[j] in self.ar_bi:
                                    next_item = item_list[j]
                                    break
                        ar_items.append(next_item)
                    assert len(ar_items) == len(item_list)
                    if len(item_list) <= self.seq_len or np.random.rand() < 0.8:
                        item_list = item_list[-self.seq_len:]
                        ar_items = ar_items[-self.seq_len:]
                        b_list = b_list[-self.seq_len:]
                        ts_list = ts_list[-self.seq_len:]
                        ar_dt = ar_dt[-self.seq_len:]
                        c_list = c_list[-self.seq_len:]

                        seq_length = len(item_list)
                        padding_len = self.seq_len - seq_length
                        item_list = item_list + [0] * padding_len
                        ar_items = ar_items + [0] * padding_len
                        b_list = b_list + [0] * padding_len
                        event_dt = ts_list + [0] * padding_len
                        ar_dt = ar_dt + [0] * padding_len
                        if self.n_attr is not None:
                            c_list = c_list + [[0] * (self.n_attr+1)] * padding_len
                        else:
                            c_list = c_list + [0] * padding_len
                    else:
                        begin_idx = np.random.randint(0, len(item_list) - self.seq_len + 1)
                        item_list = item_list[begin_idx:begin_idx + self.seq_len]
                        ar_items = ar_items[begin_idx:begin_idx + self.seq_len]
                        b_list = b_list[begin_idx:begin_idx + self.seq_len]
                        event_dt = ts_list[begin_idx:begin_idx + self.seq_len]
                        ar_dt = ar_dt[begin_idx:begin_idx + self.seq_len]
                        c_list = c_list[begin_idx:begin_idx + self.seq_len]
                        seq_length = self.seq_len
                    yield {"items": item_list, "labels": ar_items, "behaviors": b_list, "seq_length": seq_length,       # ToDo: 输出类别特征cate
                           "event_dt": event_dt, "ar_dt": ar_dt, 'cate': c_list}, [-1]*(self.seq_len-1)+[0]
                elif mode == 'eval':
                    labels = tgt_items
                    if len(labels) > 0 and flag_train != 1:
                        item_list = item_list[-self.seq_len:]
                        b_list = b_list[-self.seq_len:]
                        event_dt = ts_list[-self.seq_len:]
                        ar_dt = ar_dt[-self.seq_len:]
                        c_list = c_list[-self.seq_len:]
                        seq_length = min(self.seq_len, len(item_list))
                        padding_len = self.seq_len - len(item_list)
                        item_list = item_list + [0] * padding_len
                        b_list = b_list + [0] * padding_len
                        event_dt = event_dt + [0] * padding_len
                        labels = list(labels)[-self.seq_len:]
                        ar_items = [0] * (seq_length-1) + labels[:1] + [0]*(self.seq_len-seq_length)
                        ar_dt = ar_dt + [0] * padding_len
                        if self.n_attr is not None:
                            c_list = c_list + [[0] * (self.n_attr+1)] * padding_len
                        else:
                            c_list = c_list + [0] * padding_len
                        y = list(map(int, tgt_items))
                        y = y + [-1] * (self.seq_len - len(y))
                        yield {"items": item_list, "labels": ar_items, "behaviors": b_list, "seq_length": seq_length,
                               "event_dt": event_dt, "ar_dt": ar_dt, 'cate': c_list}, y


def main():
    with open(sys.argv[1], encoding='utf-8') as fr:
        config = yaml.load(fr.read(), Loader=yaml.Loader)
    config['data']['debug'] = True
    config['data']['seq_len'] = 10
    config['data']['target_b'] = 'pos'
    config['data']['mb'] = "{'neg': 1, 'mid': 2, 'pos': 3, 'tip': 4}"
    # config['data']['target_b'] = '2'
    # config['data']['mb'] = "{'1': 1, '2': 2, '3': 3, '4': 4, '5': 5}"
    # config['data']['mask_bi'] = "[1, 2, 3, 4, 5]"
    config['data']['n_attr'] = 18
    config['data']['file_name'] = '../benchmark/ml10_5core_idx'  # yelp_city_5core_idx'
    gs = SasRecSample(config)
    for mode in ['eval']:
        sn = 0
        for x, y in gs.gen_features(mode=mode):
            sn += 1
            # for k in ["items", "labels", "behaviors", "event_dt", "ar_dt"]:
            #     if len(x[k]) != 50:
            #         print(x)
            # if sn % 113 == 0:
            #     print(x, y)
        print(mode, sn)
    print("End")


if __name__ == "__main__":
    main()
