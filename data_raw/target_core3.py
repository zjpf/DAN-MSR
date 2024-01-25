import csv
from dateutil.parser import parse
csv.field_size_limit(1024*1024*1024)


class GenDataRaw:
    def __init__(self, start_ts=None, end_ts=None, data_type='ali'):
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.n1 = 0

    # 输入： 按用户和时间排好序的数据
    # 输出：1. 过滤非时间周期内的样本，2. 格式化：item_id#cate_id#event_type#time
    def convert(self, raw_file, standard_file, data_type):
        if data_type == 'ali':
            user_set = set()
            with open(standard_file, 'w') as fw:
                with open(raw_file) as f:
                    prior_user = '-1'
                    prior_time = 0
                    item_time_tps = []
                    for row in csv.reader(f):
                        if int(row[-1]) < self.start_ts or int(row[-1]) > self.end_ts:
                            print("continue: " + str(row))
                            self.n1 += 1
                            continue
                        if row[0] != prior_user and prior_user != '-1':
                            fw.write(prior_user + ',' + "|".join(item_time_tps) + '\n')
                            item_time_tps = []
                            prior_time = 0
                            if prior_user in user_set:
                                print(row)
                            user_set.add(prior_user)
                        item_time_tps.append('#'.join([row[1], row[3], row[2], row[-1]]))
                        prior_user = row[0]
                        if prior_time > int(row[-1]):
                            print(row)
                        prior_time = int(row[-1])
                    fw.write(prior_user + ',' + "|".join(item_time_tps) + '\n')
        elif data_type == 'jd':
            # 读取sku_id和cate_id的映射关系： cate_id做好编码
            cate_id, item_cate, i = {}, {}, 0
            with open(raw_file+'/item.csv', 'r', encoding='utf-8-sig') as f:
                for line in f.readlines():
                    row = line.strip().split(',')
                    item, c = row[0], row[3]
                    # if c in cate_id:
                    #     ci = cate_id[c]
                    # else:
                    #     i += 1
                    #     cate_id[c] = i
                    #     ci = i
                    # item_cate[item] = ci
                    item_cate[item] = c
            # 读取action关联cate_id
            user_set = set()
            with open(standard_file, 'w', encoding='utf-8') as fw:
                with open(raw_file+'/sorted_action', encoding='utf-8-sig') as f:
                    prior_user = '-1'
                    prior_time = 0
                    item_time_tps = []
                    for row in csv.reader(f):
                        ts = int(parse(row[2]).timestamp())
                        if not '2018-02-01' <= row[2][:10] <= '2018-04-15':
                            print("continue: " + str(row))
                            self.n1 += 1
                            continue
                        if row[0] != prior_user and prior_user != '-1':
                            fw.write(prior_user + ',' + "|".join(item_time_tps) + '\n')
                            item_time_tps = []
                            prior_time = 0
                            if prior_user in user_set:
                                print(row)
                            user_set.add(prior_user)
                        item_time_tps.append('#'.join([row[1], row[-1], item_cate.get(row[1], '0'), str(ts)]))
                        prior_user = row[0]
                        if prior_time > ts:
                            print(row)
                        prior_time = ts
                    fw.write(prior_user + ',' + "|".join(item_time_tps) + '\n')
        elif data_type == 'yelp':
            user_set = set()
            with open(standard_file, 'w') as fw:
                with open(raw_file) as f:
                    prior_user = '-1'
                    prior_time = '1900'
                    item_time_tps = []
                    for row in csv.reader(f):
                        # if int(row[-1]) < self.start_ts or int(row[-1]) > self.end_ts:
                        #     print("continue: " + str(row))
                        #     self.n1 += 1
                        #     continue
                        if row[0] != prior_user and prior_user != '-1':
                            fw.write(prior_user + ',' + "|".join(item_time_tps) + '\n')
                            item_time_tps = []
                            prior_time = '1900'
                            if prior_user in user_set:
                                print('p1', row)
                            user_set.add(prior_user)
                        item_time_tps.append('#'.join([row[1], row[2], row[3], row[-1]]))
                        prior_user = row[0]
                        if prior_time > row[-1] and prior_time != '1900':
                            print('p2', prior_time, row)
                        prior_time = row[-1]
                    fw.write(prior_user + ',' + "|".join(item_time_tps) + '\n')

    @staticmethod
    # 1. 先卡>=3个buy的用户；统计item购买用户集合；再卡item购买用户数>=3;
    # 2. 再卡用户购买item数>=3;
    # 3. loop: 再卡item至少1个用户购买，用户购买item数>=2
    # 4. 在过滤后的item和user集合中，且用户行为次数>=5
    def core_5_rep(fn='../taobao', fn_out=None, u_top=3, i_top=3, target='buy'):
        core5_user = set()
        item_user = {}
        ui = 0
        with open(fn, encoding='utf-8') as f:
            for row in csv.reader(f):
                tps = row[1].split('|')
                tn = 0
                if len(tps) < 5:
                    continue
                for i in range(len(tps)):
                    item, event, cate, ts = tps[i].split('#')
                    if event in [target]:   # 'pv', 'fav', 'cart',
                        tn += 1
                if tn >= u_top:
                    core5_user.add(row[0])
                    for i in range(len(tps)):
                        item, event, cate, ts = tps[i].split('#')
                        if event in [target]:  # 'pv', 'fav', 'cart',
                            ov = item_user.setdefault(item, set())
                            ov.add(row[0])
                    ui += 1
        print("user_num", len(core5_user))
        item_index = {}
        user_item = {}
        i = 1
        for tp in item_user.items():
            if len(tp[1]) >= i_top:
                item_index[tp[0]] = i
                i += 1
                for u in tp[1]:
                    ov = user_item.setdefault(u, set())
                    ov.add(tp[0])
        print("item index", i)
        flt_item_user = {}
        for tp in user_item.items():
            if len(tp[1]) >= u_top:
                for i in tp[1]:
                    ov = flt_item_user.setdefault(i, set())
                    ov.add(tp[0])
        while len(item_user) > len(flt_item_user):
            print(len(item_user), len(flt_item_user))
            item_user = flt_item_user
            item_index = {}
            user_item = {}
            i = 1
            for tp in item_user.items():
                if len(tp[1]) >= i_top:
                    item_index[tp[0]] = i
                    i += 1
                    for u in tp[1]:
                        ov = user_item.setdefault(u, set())
                        ov.add(tp[0])
            print("item index", i)
            flt_item_user = {}
            for tp in user_item.items():
                if len(tp[1]) >= u_top:
                    for i in tp[1]:
                        ov = flt_item_user.setdefault(i, set())
                        ov.add(tp[0])
            print(len(item_user), len(flt_item_user), len(user_item))

        # 循环user和item
        with open(fn_out, 'w', encoding='utf-8') as fw:
            with open(fn, encoding='utf-8') as f:
                for row in csv.reader(f):
                    tps = row[1].split('|')
                    if row[0] in user_item:
                        core_tps = []
                        for i in range(len(tps)):
                            item, event, cate, ts = tps[i].split('#')
                            if item in item_index:
                                core_tps.append('#'.join([str(item_index[item]), event, cate, ts]))
                        if len(core_tps) >= u_top:
                            fw.write(row[0] + ',' + "|".join(core_tps)+'\n')
                        # else:
                        #     print(row)

    @staticmethod
    def idx_cate(fn, fn_out):
        cate_index, cate_cnt = {}, {}
        ind = 0
        with open(fn) as f:
            for row in csv.reader(f):
                tps = row[1].split('|')
                for i in range(len(tps)):
                    item, event, cate, ts = tps[i].split('#')
                    if cate not in cate_index:
                        cate_index[cate] = str(ind)
                        ind += 1
                    ov = cate_cnt.setdefault(cate, 0)
                    cate_cnt[cate] = ov+1
        print(ind, len(cate_index))
        with open(fn_out, 'w') as fw:
            with open(fn) as f:
                for row in csv.reader(f):
                    tps = row[1].split('|')
                    out_tps = []
                    for i in range(len(tps)):
                        item, event, cate, ts = tps[i].split('#')
                        out_tps.append('#'.join([item, event, cate_index.get(cate), ts]))
                    fw.write(row[0] + ',' + "|".join(out_tps) + '\n')


if __name__ == "__main__":
    # 1. taobao
    # gd = GenDataRaw(start_ts=1511452800, end_ts=1512316799)
    # gd.convert(raw_file='D:/paper/multibehavior/sortedUserBehavior.csv', standard_file='d:/paper/multibehavior/seq_mb/benchmark/taobao', data_type='ali')
    # print(gd.n1)
    # gd.core_5_rep(fn='d:/paper/multibehavior/seq_mb/benchmark/taobao', fn_out='d:/paper/multibehavior/seq_mb/benchmark/taobao_5core_idx_v2', u_top=3, i_top=3)
    # gd.idx_cate(fn='d:/paper/multibehavior/seq_mb/benchmark/taobao_5core_idx_v2', fn_out='d:/paper/multibehavior/seq_mb/benchmark/taobao_5core_v2')
    # 2. jd
    # gd = GenDataRaw()
    # gd.convert(raw_file='D:/paper/multibehavior/raw_jdata', standard_file='d:/paper/multibehavior/seq_mb/benchmark/jdata', data_type='jd')
    # print(gd.n1)
    # gd.core_5_rep(fn='d:/paper/multibehavior/seq_mb/benchmark/jdata',
    #               fn_out='d:/paper/multibehavior/seq_mb/benchmark/jdata_5core_idx_v2', u_top=3, i_top=3, target='2')
    # gd.core_5_rep(fn='d:/paper/multibehavior/seq_mb/benchmark/jdata_5core_idx_v2',
    #               fn_out='d:/paper/multibehavior/seq_mb/benchmark/jdata_5core_idx_v3', u_top=3, i_top=3, target='2')

    # gd.idx_cate(fn='d:/paper/multibehavior/seq_mb/benchmark/jdata_5core_idx_v2', fn_out='d:/paper/multibehavior/seq_mb/benchmark/tmp')

    # 3. yelp
    gd = GenDataRaw()
    # gd.convert(raw_file='D:/paper/multibehavior/archive/sort_yelp_raw_city', standard_file='d:/paper/multibehavior/seq_mb/benchmark/yelp_city', data_type='yelp')
    # print(gd.n1)
    # gd.core_5_rep(fn='d:/paper/multibehavior/seq_mb/benchmark/yelp_city',
    #               fn_out='d:/paper/multibehavior/seq_mb/benchmark/yelp_city_5core_idx_v2', u_top=10, i_top=10, target='pos')

    # gd.convert(raw_file='D:/paper/multibehavior/archive/sort_yelp_raw_stars',
    #            standard_file='d:/paper/multibehavior/seq_mb/benchmark/yelp_stars', data_type='yelp')
    # print(gd.n1)
    # gd.core_5_rep(fn='d:/paper/multibehavior/seq_mb/benchmark/yelp_stars',
    #               fn_out='d:/paper/multibehavior/seq_mb/benchmark/yelp_stars_5core_idx', u_top=10, i_top=10, target='pos')
    # gd.idx_cate(fn='d:/paper/multibehavior/seq_mb/benchmark/yelp_stars_5core_idx',
    #             fn_out='d:/paper/multibehavior/seq_mb/benchmark/tmp')
    # 4. movie-len
    # gd.convert(raw_file='D:/paper/data/ml-25m/ml-25m/sort_ml25_raw', standard_file='d:/paper/multibehavior/seq_mb/benchmark/ml25', data_type='yelp')
    # print(gd.n1)
    # gd.core_5_rep(fn='d:/paper/multibehavior/seq_mb/benchmark/ml25',
    #               fn_out='d:/paper/multibehavior/seq_mb/benchmark/ml25_5core_idx', u_top=10, i_top=10, target='pos')
    # 5. beauty
    # gd.convert(raw_file='D:/paper/data/sort_sports_raw', standard_file='d:/paper/multibehavior/seq_mb/benchmark/sports', data_type='yelp')
    # print(gd.n1)
    # gd.core_5_rep(fn='d:/paper/multibehavior/seq_mb/benchmark/sports',
    #               fn_out='d:/paper/multibehavior/seq_mb/benchmark/sports_5core_idx', u_top=3, i_top=3, target='pos')
    # 6. ijcai
    # gd.convert(raw_file='D:/paper/multibehavior/data_format1/sort_ijcai_raw', standard_file='d:/paper/multibehavior/seq_mb/benchmark/ijcai_2015',
    #            data_type='yelp')
    # print(gd.n1)
    # gd.core_5_rep(fn='d:/paper/multibehavior/seq_mb/benchmark/ijcai_2015',
    #               fn_out='d:/paper/multibehavior/seq_mb/benchmark/ijcai_2015_5core_idx', u_top=5, i_top=5, target='2')
    gd.idx_cate(fn='d:/paper/multibehavior/seq_mb/benchmark/ijcai_2015_10core_idx',
                fn_out='d:/paper/multibehavior/seq_mb/benchmark/tmp')
    # 7. tencent
    # gd.convert(raw_file='D:/paper/data/Tenrec/Tenrec/tencent_raw_v2', standard_file='d:/paper/multibehavior/seq_mb/benchmark/tencent_v2',
    #            data_type='yelp')
    # print(gd.n1)
    # gd.core_5_rep(fn='d:/paper/multibehavior/seq_mb/benchmark/tencent_v2',
    #               fn_out='d:/paper/multibehavior/seq_mb/benchmark/tencent_5core_idx_v2', u_top=5, i_top=5, target='3')

    # 8. ml10
    # gd.convert(raw_file='D:/paper/data/ml-10M100K/sorted_ml10_raw', standard_file='d:/paper/multibehavior/seq_mb/benchmark/ml10', data_type='yelp')
    # print(gd.n1)
    # gd.core_5_rep(fn='d:/paper/multibehavior/seq_mb/benchmark/ml10',
    #               fn_out='d:/paper/multibehavior/seq_mb/benchmark/ml10_5core_idx', u_top=10, i_top=10, target='pos')
    #
    print('End')


# 108550 108550 160674
# 3846 3846
# jdata item: 25670, cate:
# yelp item: 117872, city: 1416
# cat yelp_raw_stars| sort -t',' -k1,1 -k5,5 > sort_yelp_raw_stars
# cat yelp_raw | sort -t',' -k1,1 -k5,5 > sort_yelp_raw_city

# jdata_5core_idx_v2: item=19898 69147, cate [0, 81];
# taobao_5core_v2: item=77027 141010, cate [0, 3312]
# yelp_stars_5core_idx: item=34474 54632, cate [0, 10]
# yelp_city_5core_idx_v2: item=34474 54632, cate 1416
# jdata_5core_idx_v3：18721 62542
# ml25_5core_idx: 15407 154058
# sports: 35942 59912
# ijcai-10core: 12246 23831
# tencent_5core_idx: 33453 67335 [33453 67335]
# ml10_10core_idx: 7794 66015
