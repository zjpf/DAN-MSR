import csv
import tensorflow as tf
import random
import numpy as np
import datetime
import yaml
import sys


class MlmSample:
    def __init__(self, conf):
        self.conf = conf
        self.train_start_dt = datetime.datetime.strptime(conf['data']['train_start_dt'], '%Y%m%d').date()  # yyyyMMdd格式整数
        self.train_end_dt = datetime.datetime.strptime(conf['data']['train_end_dt'], '%Y%m%d').date()
        self.eval_start_dt = datetime.datetime.strptime(conf['data']['eval_start_dt'], '%Y%m%d').date()
        self.eval_end_dt = datetime.datetime.strptime(conf['data']['eval_end_dt'], '%Y%m%d').date()
        self.standard_file = conf['data']['file_name']
        self.target_b = conf['data']['target_b']
        self.mb = eval(conf['data']['mb'])
        self.mask_prob = conf['data']['mask_prob']
        self.mask_bi = eval(conf['data']['mask_bi'])
        self.num_items = conf['data']['num_items']
        self.num_cate = conf['data'].get('num_cate', 3846)
        self.seq_len = conf['data']['seq_len']
        self.debug = conf['data']['debug']
        self.x = []
        self.y = []
        self.item_index = {}
        # self.parse_id_map()
        self.train_t0_t1 = conf['train']['train_t0_t1']
        self.eval_t0_t1 = conf['train']['eval_t0_t1']
        self.max_n_node = self.seq_len
        self.b_gnn = conf['train'].get('b_gnn', False)
        self.f_rp = conf['train'].get('b_rp', False)
        self.f_sess = conf['train'].get('b_sess', False)
        self.n_mb = len(self.mb)

    def parse_id_map(self):
        item_uv = {}  # 计算item覆盖的uv
        tn = 0.0
        with open(self.standard_file) as f:
            for row in csv.reader(f):
                items = set()
                for tp in row[1].split('|'):
                    item, event, ts = tp.split('#')
                    items.add(item)
                for item in items:
                    ov = item_uv.setdefault(item, 0)
                    item_uv[item] = ov + 1
                tn += len(items)
        item_uv = sorted(item_uv.items(), key=lambda x: x[1], reverse=True)
        item_prob = {}
        i = 1
        for tp in item_uv:
            if tp[1] >= 0:
                self.item_index[tp[0]] = i
                item_prob[i] = tp[1] / tn
                i += 1
        item_list = list(item_prob.keys())
        print("item_list size: {}".format(len(item_list)))

    def get_f_adj(self, masked_items, seq_length, ts_list):
        id_items, f_ids = [], []
        n_node = 0
        nodes = {}
        for it in masked_items[:seq_length]:
            ind = nodes.get(it, -1)
            if it == self.num_items + 1:
                id_items.append(it)
                ind = n_node
                n_node += 1
            elif ind < 0:
                id_items.append(it)
                nodes[it] = n_node
                ind = n_node
                n_node += 1
            f_ids.append(ind)
        if n_node > self.max_n_node:
            self.max_n_node = n_node
        u_A = np.zeros((self.max_n_node, self.max_n_node))
        for i in range(seq_length-1):
            u = f_ids[i]
            v = f_ids[i+1]
            if ts_list[i+1]-ts_list[i] < 10.8:  # 限制30分钟，捕捉短期关系
                u_A[u][v] += 1
        u_sum_in = np.sum(u_A, 0)
        u_sum_in[np.where(u_sum_in == 0)] = 1
        u_A_in = np.divide(u_A, u_sum_in)

        u_sum_out = np.sum(u_A, 1)
        u_sum_out[np.where(u_sum_out == 0)] = 1
        u_A_out = np.divide(u_A.transpose(), u_sum_out)
        f_ids = f_ids + [0] * (self.seq_len-len(f_ids))
        id_items = id_items + [0] * (self.seq_len-len(id_items))
        return {'f_ids': f_ids, 'f_id_items': id_items, 'f_ain': u_A_in, 'f_aout': u_A_out}

    def get_gid(self, i1, b1):
        return i1*self.n_mb+(b1-1)

    def get_f_adj_mb(self, masked_items, b_list, seq_length, ts_list):
        id_items, id_bhv, f_ids = [0] * (self.seq_len*self.n_mb), [0] * (self.seq_len*self.n_mb), []
        n_node = 0
        nodes = {}      # 保存item对应的初始id
        for it, b in zip(masked_items[:seq_length], b_list[:seq_length]):
            ind = nodes.get(it, -1)
            if it == self.num_items + 1:
                ind = n_node
                n_node += 1
            elif ind < 0:
                nodes[it] = n_node
                ind = n_node
                n_node += 1
            gid = self.get_gid(ind, b)
            id_items[gid] = it
            id_bhv[gid] = b
            f_ids.append(gid)
        if n_node > self.max_n_node:
            self.max_n_node = n_node
        u_A = np.zeros((self.max_n_node*self.n_mb, self.max_n_node*self.n_mb))
        for i in range(seq_length-1):
            u = f_ids[i]
            v = f_ids[i+1]
            if ts_list[i+1]-ts_list[i] < 10.8:  # 限制30分钟，捕捉短期关系
                u_A[u][v] += 1
        u_sum_in = np.sum(u_A, 0)
        u_sum_in[np.where(u_sum_in == 0)] = 1
        u_A_in = np.divide(u_A, u_sum_in)

        u_sum_out = np.sum(u_A, 1)
        u_sum_out[np.where(u_sum_out == 0)] = 1
        u_A_out = np.divide(u_A.transpose(), u_sum_out)
        f_ids = f_ids + [0] * (self.seq_len-len(f_ids))
        # id_items = id_items + [0] * (self.seq_len-len(id_items))
        return {'f_ids': f_ids, 'f_id_items': id_items, 'f_ain': u_A_in, 'f_aout': u_A_out, 'f_id_bhv': id_bhv}

    def get_f_rp(self, b_seq):
        rp = []
        for i in range(len(b_seq)):
            v = [0] * len(b_seq)
            b_af = [0] * (self.n_mb + 1)
            for j in range(i + 1, len(b_seq)):
                bj = b_seq[j]
                b_af[bj] += 1
                v[j] = b_af[bj]
            b_bf = [0] * (self.n_mb + 1)
            for j in range(i - 1, -1, -1):
                bj = b_seq[j]
                b_bf[bj] -= 1
                v[j] = b_bf[bj]
            rp.append(v)
        return {'f_rp': rp}

    @staticmethod
    def get_f_sess(ts_list):
        f_sess = [0]
        si = 0
        p_ts = ts_list[0]
        for i in range(1, len(ts_list)):
            if ts_list[i]-p_ts > 30*60:
                si += 1
            f_sess.append(si)
            p_ts = ts_list[i]
        return {'f_sess': f_sess}

    # 产出包括cate_id
    def gen_features(self, mode='train'):
        with open(self.standard_file) as f:
            for row in csv.reader(f):
                item_list, b_list, ts_list, dt_list, c_list = [], [], [], [], []
                tps = row[1].split('|')
                for i in range(len(tps)):
                    item, event, cate, ts = tps[i].split('#')
                    if len(ts) == 10:
                        dt = datetime.datetime.fromtimestamp(int(ts)).date()
                    elif self.standard_file.find('ijcai') > 0 or self.standard_file.find('yelp'):
                        dt = self.train_start_dt
                    else:
                        continue
                    if (mode == 'train' and self.train_t0_t1 == 'T0') or self.train_start_dt <= dt <= self.train_end_dt or (
                            mode == 'eval' and self.eval_t0_t1 in ['T0', 'T0_last']):
                        if self.debug or len(self.item_index) == 0:
                            item_list.append(int(item))
                            c_list.append(int(cate))
                        else:
                            item_list.append(self.item_index.get(item))
                        b_list.append(self.mb.get(event))
                        try:
                            ts_list.append(int(ts))
                        except ValueError:
                            ts_list.append(0)
                        dt_list.append(dt)
                if len(item_list) < 3:
                    continue
                if mode == 'train':
                    masked_items, masked_labels, masked_c = [], [], []
                    if self.train_t0_t1 == 'T0' and b_list[-1] == self.mb[self.target_b]:
                        item_list = item_list[:-1]
                        c_list = c_list[:-1]
                        b_list = b_list[:-1]
                        ts_list = ts_list[:-1]
                    for s, b, c in zip(item_list, b_list, c_list):
                        prob = np.random.rand()
                        if prob < self.mask_prob and b in self.mask_bi:
                            masked_items.append(self.num_items + 1)
                            masked_labels.append(s)
                            masked_c.append(self.num_cate+1)    # ToDO: 类别特征和item同时mask
                        else:
                            masked_items.append(s)
                            masked_labels.append(0)
                            masked_c.append(c)

                    if len(masked_items) <= self.seq_len or np.random.rand() < 0.8:
                        masked_items = masked_items[-self.seq_len:]
                        masked_labels = masked_labels[-self.seq_len:]
                        b_list = b_list[-self.seq_len:]
                        ts_list = ts_list[-self.seq_len:]
                        masked_c = masked_c[-self.seq_len:]

                        seq_length = len(masked_items)
                        padding_len = self.seq_len - len(masked_items)
                        masked_items = masked_items + [0] * padding_len
                        masked_labels = masked_labels + [0] * padding_len
                        b_list = b_list + [0] * padding_len
                        ts_list = ts_list + [0.] * padding_len
                        masked_c = masked_c + [0] * padding_len
                    else:
                        begin_idx = np.random.randint(0, len(masked_items) - self.seq_len + 1)
                        masked_items = masked_items[begin_idx:begin_idx + self.seq_len]
                        masked_labels = masked_labels[begin_idx:begin_idx + self.seq_len]
                        b_list = b_list[begin_idx:begin_idx + self.seq_len]
                        ts_list = ts_list[begin_idx:begin_idx + self.seq_len]
                        masked_c = masked_c[begin_idx:begin_idx + self.seq_len]
                        seq_length = self.seq_len
                    f_t = {"items": masked_items, "labels": masked_labels, "behaviors": b_list, "seq_length": seq_length, "event_dt": ts_list, 'cate': masked_c}
                    if self.b_gnn == 'sb':
                        f_t.update(self.get_f_adj(masked_items, seq_length, ts_list))
                    elif self.b_gnn == 'mb':
                        f_t.update(self.get_f_adj_mb(masked_items, b_list, seq_length, ts_list))
                    if self.f_rp:
                        f_t.update(self.get_f_rp(b_list))
                    if self.f_sess:
                        f_t.update(self.get_f_sess(ts_list))
                    yield f_t, 0
                elif mode == 'eval':
                    if b_list[-1] == self.mb[self.target_b]:
                        label = item_list[-1:]
                        item_list[-1] = self.num_items + 1
                        item_list = item_list[-self.seq_len:]
                        b_list = b_list[-self.seq_len:]
                        ts_list = ts_list[-self.seq_len:]
                        c_list[-1] = self.num_cate+1
                        c_list = c_list[-self.seq_len:]
                        seq_length = min(self.seq_len, len(item_list))
                        padding_len = self.seq_len - len(item_list)
                        item_list = item_list + [0] * padding_len
                        b_list = b_list + [0] * padding_len
                        ts_list = ts_list + [0.] * padding_len
                        c_list = c_list + [0] * padding_len
                        labels = [0]*(seq_length-1) + label + [0] * (self.seq_len-seq_length)
                        f_t = {"items": item_list, "labels": labels, "behaviors": b_list, "seq_length": seq_length, "event_dt": ts_list, 'cate': c_list}
                        if self.b_gnn == 'sb':
                            f_t.update(self.get_f_adj(item_list, seq_length, ts_list))
                        elif self.b_gnn == 'mb':
                            f_t.update(self.get_f_adj_mb(item_list, b_list, seq_length, ts_list))
                        if self.f_rp:
                            f_t.update(self.get_f_rp(b_list))
                        if self.f_sess:
                            f_t.update(self.get_f_sess(ts_list))
                        yield f_t, label[0]


def main():
    with open(sys.argv[1], encoding='utf-8') as fr:
        config = yaml.load(fr.read(), Loader=yaml.Loader)
    config['data']['debug'] = True
    config['data']['seq_len'] = 6
    config['train']['b_gnn'] = 'mb'
    config['train']['train_t0_t1'] = 'T0'
    config['data']['file_name'] = '../benchmark/taobao_5core_idx'
    gs = MlmSample(config)
    for mode in ['eval']:
        sn = 0
        for x, y in gs.gen_features(mode=mode):
            sn += 1
            if sn % 113 == 0:
                print(x, y)
        print(mode, sn)
    print('max_n_node', gs.max_n_node)
    print("End")


if __name__ == "__main__":
    main()
