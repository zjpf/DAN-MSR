import csv
import numpy as np
import datetime
import yaml
import sys
csv.field_size_limit(1024*1024*1024)


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
        self.train_t0_t1 = conf['train']['train_t0_t1']
        self.eval_t0_t1 = conf['train']['eval_t0_t1']
        self.n_mb = len(self.mb)
        self.n_attr = conf['data'].get('n_attr', None)
        # 样本统计信息：跳过的样本，co_buy的分布
        self.skip_sn = 0
        self.co_buy_cnt = {}

    def check_len(self, ft):
        for v in ["items", "labels", "behaviors", 'cate', 'event_dt', 'cate']:
            if len(ft[v]) != self.seq_len:
                print(ft)

    # 产出包括cate_id
    def gen_features(self, mode='train'):
        with open(self.standard_file) as f:
            for row in csv.reader(f):
                item_list, b_list, ts_list, dt_list, c_list = [], [], [], [], []
                tps = row[1].split('|')
                # 1. 取最后一次行为为目标行为，再取和最后一次行为发生时间相同的目标行为，作为labels
                i, co_buy, tgt_items, prior_ts, flag_train = -1, 0, set(), -1, 0
                while i+len(tps) >= 0:
                    item, event, cate, ts = tps[i].split('#')
                    if prior_ts == ts or prior_ts == -1:  # or self.standard_file.find('ijcai') > 0:  # ijcai特殊处理，取最后一天行为为label
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
                # if co_buy == len(tps) or len(tps)-co_buy < 3:   # or len(tps)-co_buy < 3  训练item数小于3，则将该样本所有item加入训练集
                #     co_buy = 0
                #     flag_train = 1  # 直接忽略leave-one-out训练长度<3的部分
                #     # continue
                if len(tps)-co_buy <= 0:    # 2. 当全部为目标行为时，跳过样本
                    # self.skip_sn += 1
                    continue
                # ov = self.co_buy_cnt.setdefault(co_buy, 0)
                # self.co_buy_cnt[co_buy] = ov + 1
                if co_buy > 0:      # 3. 将随机一行为加入作为surrogate目标：co_buy行为过多时，surrogate目标越不准确
                    co_buy = co_buy - 1
                for i in range(len(tps)-co_buy):
                    item, event, cate, ts = tps[i].split('#')
                    if len(ts) == 10:
                        dt = datetime.datetime.fromtimestamp(int(ts)).date()
                    elif self.standard_file.find('ijcai') > 0 or self.standard_file.find('yelp'):
                        dt = self.train_start_dt
                    else:
                        continue
                    if (mode == 'train' and self.train_t0_t1 == 'T0') or self.train_start_dt <= dt <= self.train_end_dt or (
                            mode == 'eval' and self.eval_t0_t1 in ['T0', 'T0_last']):
                        if self.debug or len(self.item_index) == 0:     # 4. 默认数据集中item和cate都已索引好
                            item_list.append(int(item))
                            if self.n_attr is not None:     # 5. 若配置了n_attr，表示是multi-hot的属性输入
                                attr_c = [0]*(self.n_attr+1)
                                for attr in cate.split('^'):
                                    if len(attr) > 0:
                                        attr_c[int(attr)] = 1
                                c_list.append(attr_c)
                            else:
                                c_list.append(int(cate))
                        else:
                            item_list.append(self.item_index.get(item))
                        b_list.append(self.mb.get(event))
                        try:
                            ts_list.append(int(ts))
                        except ValueError:
                            ts_list.append(0)
                        dt_list.append(dt)
                # if len(item_list) < 3:  #
                # if (b_list[-1] == self.mb[self.target_b] and len(item_list) <= 2) or (
                #      b_list[-1] != self.mb[self.target_b] and len(item_list) <= 1):
                #     print(row)
                    # continue
                if mode == 'train':
                    masked_items, masked_labels, masked_c = [], [], []
                    if self.train_t0_t1 == 'T0' and len(tgt_items) > 0 and flag_train != 1:   # 6. 最后一次为目标行为，为label，ToDo:[会多加入非目标行为label为训练item]
                        item_list = item_list[:-1]
                        c_list = c_list[:-1]
                        b_list = b_list[:-1]
                        ts_list = ts_list[:-1]
                    for s, b, c in zip(item_list, b_list, c_list):
                        prob = np.random.rand()
                        if prob < self.mask_prob and b in self.mask_bi:
                            masked_items.append(self.num_items + 1)
                            masked_labels.append(s)
                            if self.n_attr is not None:
                                masked_c.append([0]*self.n_attr+[1])
                            else:
                                masked_c.append(self.num_cate+1)
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
                        if self.n_attr is not None:
                            masked_c = masked_c + [[0] * (self.n_attr+1)] * padding_len
                        else:
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
                    yield f_t, [-1]*(self.seq_len-1)+[0]
                elif mode == 'eval':
                    if len(tgt_items) > 0 and flag_train != 1:  # 7. 有目标label作为测试集，最后一个是surrogate label
                        label = item_list[-1:]
                        item_list[-1] = self.num_items + 1
                        item_list = item_list[-self.seq_len:]
                        b_list[-1] = self.mb[self.target_b]     # ToDo: ijcai数据集存在最后一个surrogate label不是target label
                        b_list = b_list[-self.seq_len:]
                        ts_list = ts_list[-self.seq_len:]
                        if self.n_attr is not None:
                            c_list[-1] = [0] * self.n_attr + [1]
                        else:
                            c_list[-1] = self.num_cate+1
                        c_list = c_list[-self.seq_len:]
                        seq_length = min(self.seq_len, len(item_list))
                        padding_len = self.seq_len - len(item_list)
                        item_list = item_list + [0] * padding_len
                        b_list = b_list + [0] * padding_len
                        ts_list = ts_list + [0.] * padding_len
                        if self.n_attr is not None:
                            c_list = c_list + [[0] * (self.n_attr+1)] * padding_len
                        else:
                            c_list = c_list + [0] * padding_len
                        labels = [0]*(seq_length-1) + label + [0] * (self.seq_len-seq_length)
                        f_t = {"items": item_list, "labels": labels, "behaviors": b_list, "seq_length": seq_length, "event_dt": ts_list, 'cate': c_list}
                        # self.check_len(f_t)
                        y = list(map(int, tgt_items))
                        y = y + [-1]*(self.seq_len-len(y))
                        yield f_t, y  # '^'.join(tgt_items)


def main():
    with open(sys.argv[1], encoding='utf-8') as fr:
        config = yaml.load(fr.read(), Loader=yaml.Loader)
    config['data']['debug'] = True
    config['data']['seq_len'] = 100
    config['data']['seq_len'] = 100
    # config['train']['b_gnn'] = 'mb'
    config['train']['train_t0_t1'] = 'T0'
    config['data']['target_b'] = 'pos'
    config['data']['mb'] = "{'neg': 1, 'mid': 2, 'pos': 3, 'tip': 4}"
    # config['data']['mask_bi'] = "[1, 2, 3, 4, 5]"
    # config['data']['n_attr'] = 19
    config['data']['target_b'] = '3'
    config['data']['mb'] = "{'0': 1, '1': 2, '2': 3, '3': 4}"  # "{'1': 1, '2': 2, '3': 3, '4': 4, '5': 5}"
    # config['data']['target_b'] = 'buy'
    # config['data']['n_attr'] = 18
    config['data']['file_name'] = '../benchmark/tencent_5core_idx'  # yelp_stars_5core_idx'
    gs = MlmSample(config)
    for mode in ['eval']:
        sn = 0
        for x, y in gs.gen_features(mode=mode):
            sn += 1
            # if sn % 1130 == 0:
            #     print(x, y)
        print(mode, sn)
    print('max_n_node', gs.max_n_node)
    print("End")


if __name__ == "__main__":
    main()
