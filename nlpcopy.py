import re
import json

from tensorflow import keras
from tqdm import tqdm
import jieba
import codecs
import editdistance
import pandas as pd

from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.layers import *
from keras.models import Model,Sequential
import keras.backend as K

from keras.callbacks import Callback
import numpy as np
import tensorflow as tf

gpus = tf.config.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)

def read_data(data_file, table_file):
    data, tables = [], {}
    with open(data_file,'r',encoding='utf-8') as f:
        for l in f:
            data.append(json.loads(l))
    with open(table_file,'r',encoding='utf8') as f:
        for l in f:
            l = json.loads(l)
            d = {}
            d['headers'] = l['header']
            d['header2id'] = {j: i for i, j in enumerate(d['headers'])}
            d['content'] = {}
            d['all_values'] = set()
            rows = np.array(l['rows'])
            for i, h in enumerate(d['headers']):
                d['content'][h] = set(rows[:, i])
                d['all_values'].update(d['content'][h])
            d['all_values'] = set([i for i in d['all_values'] if hasattr(i, '__len__')])
            tables[l['id']] = d
    return data, tables


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R


def seq_padding(X, padding=0, maxlen=None):
    if maxlen is None:
        L = [len(x) for x in X]
        ML = max(L)
    else:
        ML = maxlen
    return np.array([
        np.concatenate([x[:ML], [padding] * (ML - len(x))]) if len(x[:ML]) < ML else x for x in X
    ])


def most_similar(s, slist):
    """从词表中找最相近的词（当无法全匹配的时候）
    """
    if len(slist) == 0:
        return s
    scores = [editdistance.eval(s, t) for t in slist]
    return slist[np.argmin(scores)]


def most_similar_2(w, s):
    """从句子s中找与w最相近的片段，
    借助分词工具和ngram的方式尽量精确地确定边界。
    """
    sw = jieba.lcut(s)
    sl = list(sw)
    sl.extend([''.join(i) for i in zip(sw, sw[1:])])
    sl.extend([''.join(i) for i in zip(sw, sw[1:], sw[2:])])
    return most_similar(w, sl)



class data_generator:
    def __init__(self, data, tables, batch_size=16):
        self.data = data
        self.tables = tables
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs) # 随机排序下标,输入的数据不按顺序
            X1, X2, XM, H, HM, SEL, CONN, CSEL, COP = [], [], [], [], [], [], [], [], []
            for i in idxs:
                d = self.data[i] # 某一条数据的全部
                # 对question部分进行编码
                x1, x2 = tokenizer.encode(d['question'])
                # 设置qustion部分的mask
                xm = [0] + [1] * len(d['question']) + [0]
                h = []
                t = self.tables[d['table_id']]['headers']
                for j in t:
                    _x1, _x2 = tokenizer.encode(j)
                    h.append(len(x1))
                    x1.extend(_x1) # 把headers的encodeing拼接到question之后
                    x2.extend(_x2)
                hm = [1] * len(h) # header的mask

                sel = []
                for j in range(len(h)):
                    if j in d['sql']['sel']:
                        j = d['sql']['sel'].index(j)
                        sel.append(d['sql']['agg'][j])
                    else:
                        sel.append(num_agg - 1)  # 不被select则被标记为num_agg-1

                conn = [d['sql']['cond_conn_op']]
                # 非条件列标记0
                csel = np.zeros(len(d['question']) + 2, dtype='int32')
                # 不被select则被标记为num_op-1
                cop = np.zeros(len(d['question']) + 2, dtype='int32') + num_op - 1
                for j in d['sql']['conds']: # 条件值匹配，匹配不上则找最相似的
                    if j[2] not in d['question']:
                        j[2] = most_similar_2(j[2], d['question'])
                    if j[2] not in d['question']:
                        continue
                    k = d['question'].index(j[2])
                    csel[k + 1: k + 1 + len(j[2])] = j[0] # csel对应的部分->条件列
                    cop[k + 1: k + 1 + len(j[2])] = j[1] # cop对应的部分->条件类型
                if len(x1) > maxlen:
                    continue
                X1.append(x1)  # bert的输入
                X2.append(x2)  # bert的输入
                XM.append(xm)  # 输入序列的mask
                H.append(h)  # 列名所在位置
                HM.append(hm)  # 列名mask
                SEL.append(sel)  # 被select的列
                CONN.append(conn)  # 连接类型
                CSEL.append(csel)  # 条件中的列
                COP.append(cop)  # 条件中的运算符（同时也是值的标记）

                if len(X1) == self.batch_size:

                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    XM = seq_padding(XM, maxlen=X1.shape[1])
                    H = seq_padding(H)
                    HM = seq_padding(HM)
                    SEL = seq_padding(SEL)
                    CONN = seq_padding(CONN)
                    CSEL = seq_padding(CSEL)
                    COP = seq_padding(COP)

                    X1 = tf.convert_to_tensor(X1)
                    X2 = tf.convert_to_tensor(X2)
                    XM = tf.convert_to_tensor(XM)
                    H = tf.convert_to_tensor(H)
                    HM = tf.convert_to_tensor(HM)
                    SEL = tf.convert_to_tensor(SEL)
                    CONN = tf.convert_to_tensor(CONN)
                    CSEL = tf.convert_to_tensor(CSEL)
                    COP = tf.convert_to_tensor(COP)


                    inputx = [X1,X2,XM,H,HM,SEL,CONN,CSEL,COP]
                    outputy = [SEL,CONN,CSEL,COP]
                    print("generate once")
                    yield inputx,outputy
                    X1, X2, XM, H, HM, SEL, CONN, CSEL, COP = [], [], [], [], [], [], [], [], []


def seq_gather(x):
    """seq是[None, seq_len, s_size]的格式，
    idxs是[None, n]的格式，在seq的第i个序列中选出第idxs[i]个向量，
    最终输出[None, n, s_size]的向量。
    """
    seq, idxs = x
    idxs = K.cast(idxs, 'int32')
    print("seq:",seq)
    print("indx:",idxs)
    return tf.compat.v1.batch_gather(seq,idxs)


def cn2sql(question, table):
    """输入question和headers，转SQL
    """
    x1, x2 = tokenizer.encode(question)
    h = []
    for i in table['headers']:
        _x1, _x2 = tokenizer.encode(i)
        h.append(len(x1))
        x1.extend(_x1)
        x2.extend(_x2)
    hm = [1] * len(h)

    psel, pconn, pcop, pcsel = train_model.predict([
        np.array([x1]),
        np.array([x2]),
        np.array([h]),
        np.array([hm])
    ])
    R = {'agg': [], 'sel': []}
    for i, j in enumerate(psel[0].argmax(1)):
        if j != num_agg - 1:  # num_agg-1类是不被select的意思
            R['sel'].append(i)
            R['agg'].append(j)
    conds = []
    v_op = -1
    for i, j in enumerate(pcop[0, :len(question) + 1].argmax(1)):
        # 这里结合标注和分类来预测条件
        if j != num_op - 1:
            if v_op != j:
                if v_op != -1:
                    v_end = v_start + len(v_str)
                    csel = pcsel[0][v_start: v_end].mean(0).argmax()
                    conds.append((csel, v_op, v_str))
                v_start = i
                v_op = j
                v_str = question[i - 1]
            else:
                v_str += question[i - 1]
        elif v_op != -1:
            v_end = v_start + len(v_str)
            csel = pcsel[0][v_start: v_end].mean(0).argmax()
            conds.append((csel, v_op, v_str))
            v_op = -1
    R['conds'] = set()
    for i, j, k in conds:
        if re.findall('[^\d\.]', k):
            j = 2  # 非数字只能用等号
        if j == 2:
            if k not in table['all_values']:
                # 等号的值必须在table出现过，否则找一个最相近的
                k = most_similar(k, list(table['all_values']))
            h = table['headers'][i]
            # 然后检查值对应的列是否正确，如果不正确，直接修正列名
            if k not in table['content'][h]:
                for r, v in table['content'].items():
                    if k in v:
                        i = table['header2id'][r]
                        break
        R['conds'].add((i, j, k))
    R['conds'] = list(R['conds'])
    if len(R['conds']) <= 1:  # 条件数少于等于1时，条件连接符直接为0
        R['cond_conn_op'] = 0
    else:
        R['cond_conn_op'] = 1 + pconn[0, 1:].argmax()  # 不能是0
    return R


def is_equal(R1, R2):
    """判断两个SQL字典是否全匹配
    """
    return (R1['cond_conn_op'] == R2['cond_conn_op']) & \
           (set(zip(R1['sel'], R1['agg'])) == set(zip(R2['sel'], R2['agg']))) & \
           (set([tuple(i) for i in R1['conds']]) == set([tuple(i) for i in R2['conds']]))


def evaluate(data, tables):
    right = 0.
    pbar = tqdm()
    F = open('evaluate_pred.json', 'w')
    for i, d in enumerate(data):
        question = d['question']
        table = tables[d['table_id']]
        R = cn2sql(question, table)
        right += float(is_equal(R, d['sql']))
        pbar.update(1)
        pbar.set_description('< acc: %.5f >' % (right / (i + 1)))
        d['sql_pred'] = R
        s = json.dumps(d, ensure_ascii=False, indent=4)
        F.write(s + '\n')
    F.close()
    pbar.close()
    return right / len(data)


def test(data, tables, outfile='result.json'):
    pbar = tqdm()
    F = open(outfile, 'w')
    for i, d in enumerate(data):
        question = d['question']
        table = tables[d['table_id']]
        R = cn2sql(question, table)
        pbar.update(1)
        s = json.dumps(R, ensure_ascii=False)
        F.write(s + '\n')
    F.close()
    pbar.close()


def construct_train_model():
    # [1]. 输入部分
    x1_in = Input(shape=(None,), dtype='int32')  # [None, seq_len]，question和列的token_id
    x2_in = Input(shape=(None,))  # [None, seq_len]  全部为0
    xm_in = Input(shape=(None,))  # [None, seq_len], 仅question部分是1，其余是0(包括 所有的CLS和SEP，仅该部分需要标注)

    h_in = Input(shape=(), dtype='int32')  # [None, col_len]，列所在的index，递增
    hm_in = Input(shape=(None,))  # [None, col_len], 存在列的value是1，其余为0

    #下面这些是用来对比的
    sel_in = Input(shape=(None,), dtype='int32')  # [None, col_len]，6表示无选择，其余对应index
    conn_in = Input(shape=(1,), dtype='int32')    # [None, 1] 条件的连接符号? 这里为什么是1啊，不应该是3吗
    csel_in = Input(shape=(None,), dtype='int32') # [None, seq_len] 基于原始question将列的index标注上
    cop_in = Input(shape=(None,), dtype='int32')  # [None, seq_len] 基于原始question将列的index的op符号index标注上

    x1, x2, xm, h, hm,sel,conn,csel,cop= (
        x1_in, x2_in, xm_in, h_in, hm_in,sel_in,conn_in,csel_in,cop_in
    )

    print("X1-shape:",x1_in)
    # [2]. bert model get encode feature
    x = bert_model([x1_in, x2_in])  # [None, seq_len, 768]
    print("x-shape:",x)
    # [3]. 构造子任务（特征加工）
    # 第一个子任务，预测connection（and、or、空），相当于三分类
    x4conn = Lambda(lambda x: x[:, 0])(x)  # 取x1序列的第一个位置 [CLS]向量
    print("x4conn:",x4conn)
    pconn = Dense(num_cond_conn_op, activation='softmax')(x4conn)
    print("pconn:",pconn)
    # 第二个子任务，预测选择的列（列+agg），每个列的7分类
    x4h = Lambda(seq_gather)([x, h])  # 取出每个列第一个位置 [CLS]的向量
    print("x4h:",x4h)
    psel = Dense(num_agg, activation='softmax')(x4h)  # 每个列是7分类
    print("psel:",psel)
    # 第三个子任务，预测选择的op，每个seq的位置是5分类

    pcop = Dense(num_op, activation='softmax')(x)  # 直接基于原始的位置选择op，每个位置是5分类，后面需要mask（列部分不需要） [None, seq_len, 5]
    print("pcop:",pcop)
    # 第四个子任务，每个每个位置的列（col_index op col_value）,每个位置是col_len的多分类
    # 将quesition部分和列的特征进行融合
    x = Lambda(lambda x: K.expand_dims(x, 2))(x)
    print("after+1-x:",x)
   # x = Lambda(lambda x:K.expand_dims(x,3))(x)
    x4h = Lambda(lambda x: K.expand_dims(x, 1))(x4h)
    print("after+1-x4h:", x4h)
    #slice1=x4h[:,0:3]
    #slice2=x4h[:,4:5]
    #x4h=tf.concat([slice1,slice2],1)
    #test = Lambda(lambda x:x[0]+x[1])([x,x4h])

    pcsel_1 = Dense(32)(x)
    print("pcsel1:",pcsel_1)
    pcsel_2 = Dense(32)(x4h)
    print("pcsel2:",pcsel_2)
    pcsel = Add()([pcsel_1, pcsel_2])

    hm = Lambda(lambda x: K.expand_dims(x, 1))(hm)  # header的mask
    print("hm:",hm)

    pcsel = Activation('tanh')(pcsel)

    pcsel = Dense(1)(pcsel)
    print("pcsel:",pcsel)
    pcsel = Lambda(lambda x: x[0][..., 0] - (1 - x[1]) * 1e10)(
        [pcsel, hm])  # 因为col_len中pad的部分，所以要mask
    pcsel = Activation('softmax')(pcsel)

    # [4]. 组装模型
    model = Model(
        [x1_in, x2_in, h_in, hm_in],
        [psel, pconn, pcop, pcsel]
    )  # 这个应该无用的

    train_model = tf.keras.Model(
        [x1_in, x2_in, xm_in, h_in, hm_in,sel_in,conn_in,cop_in,csel_in],
        [pcsel, pconn, pcop, pcsel]
    )




    #cop_in = Lambda(lambda x: K.expand_dims(x, 1))(cop_in)
    # [5]. 添加loss
    xm = xm  # question的mask.shape=(None, seq_len)
    hm = hm[:, 0]  # header的mask.shape=(None, col_len)
      # 不等于不被选择是True，conds的mask.shape=(None, seq_len)

    # 第一个任务的loss
    # 输入 [None, 1] [None, 3]
    pconn_loss = K.sparse_categorical_crossentropy(conn_in, pconn)
    pconn_loss = K.mean(pconn_loss)  # 连接符loss，三分类

    # 第二个子任务的loss
   # psel = Flatten()(psel)
    psel_loss = K.sparse_categorical_crossentropy(sel_in, psel)
    psel_loss = K.sum(psel_loss * hm) / K.sum(hm)  # 去掉pad部分

    # 第三个任务的loss
    #pcop = Flatten()(pcop)
    #cop_in = Flatten()(cop_in)
    print("copin pcop:",cop_in,pcop)

    pcop_loss = K.sparse_categorical_crossentropy(cop_in, pcop)
    pcop_loss = K.sum(pcop_loss * xm) / K.sum(xm)  # 每个序列的多分类，排除掉CLS、[SEP]和pad等部分

    # 第四个任务的loss
    cm = K.cast(K.not_equal(cop, num_op - 1), 'float32')
    pcsel_loss = K.sparse_categorical_crossentropy(csel_in, pcsel)  # 输入 [None, seq_len], [None, seq_len, col_len]
    pcsel_loss = K.sum(pcsel_loss * cm) / K.sum(cm)  # 每个序列的多分类，只看序列部分中存在值且op有值部分

    # 四个任务的loss汇总
    loss = psel_loss + pconn_loss + pcop_loss + pcsel_loss
    train_model.add_loss(loss)

    return train_model


class Evaluate(Callback):
    def __init__(self):
        self.accs = []
        self.best = 0.
        self.passed = 0
        self.stage = 0

    def on_batch_begin(self, batch, logs=None):
        """第一个epoch用来warmup，第二个epoch把学习率降到最低
        """
        print("on batch begin---")
        if self.passed < self.params['steps']:
            lr = (self.passed + 1.) / self.params['steps'] * learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1
        elif self.params['steps'] <= self.passed < self.params['steps'] * 2:
            lr = (2 - (self.passed + 1.) / self.params['steps']) * (learning_rate - min_learning_rate)
            lr += min_learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1

    def on_epoch_end(self, epoch, logs=None):
        acc = self.evaluate()
        self.accs.append(acc)
        if acc > self.best:
            self.best = acc
            train_model.save_weights('best_model.weights')
        print('acc: %.5f, best acc: %.5f\n' % (acc, self.best))

    def evaluate(self):
        print("evaluating----")
        return evaluate(valid_data, valid_tables)


if __name__ == "__main__":

    # [1]. define variable
    maxlen = 160
    num_agg = 7  # agg_sql_dict = {0:"", 1:"AVG", 2:"MAX", 3:"MIN", 4:"COUNT", 5:"SUM", 6:"不被select"}
    num_op = 5  # {0:">", 1:"<", 2:"==", 3:"!=", 4:"不被select"}
    num_cond_conn_op = 3  # conn_sql_dict = {0:"", 1:"and", 2:"or"}
    learning_rate = 5e-5
    min_learning_rate = 1e-5


    dict_path = 'chinese_wwm_ext_L-12_H-768_A-12/vocab.txt'

    # [2]. read train & valid & test data(including question data and table data)
    train_data, train_tables = read_data('TableQA-master/train/train.json', 'TableQA-master/train/train.tables.json')
    valid_data, valid_tables = read_data('TableQA-master/val/val.json', 'TableQA-master/val/val.tables.json')
    test_data, test_tables = read_data('TableQA-master/test/test.json', 'TableQA-master/test/test.tables.json')

    # [3]. define tokenizer
    token_dict = {}
    with codecs.open(dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip() # 移除空格?
            token_dict[token] = len(token_dict)
    tokenizer = OurTokenizer(token_dict)

    # [4]. load bert model
    config_path = 'chinese_wwm_ext_L-12_H-768_A-12/bert_config.json'
    checkpoint_path = 'chinese_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

    for l in bert_model.layers:
        l.trainable = True

    # [5]. define train model
    train_model = construct_train_model()
    print("model:",train_model)
    train_model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                        loss=tf.keras.losses.SparseCategoricalCrossentropy())

    train_model.summary()
    print("description of model")
    # [6]. get data generator
    train_D = data_generator(train_data, train_tables) # input data
    print("after data generate")
    # [7]. train model
    evaluator = Evaluate()
    print("evaluate")

    x = train_D.__iter__()
    print(x)
    #tensor_dataset = tf.data.Dataset.from_tensors((x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8]))
    #ideal_dataset = tf.data.Dataset.from_tensors((y[0],y[1],y[2],y[3]))


    #dataset = tf.data.Dataset.zip((tensor_dataset,ideal_dataset)).batch(32)
    #print(tensor_dataset)

    #print(tensor_dataset.__class__)


    print("start-to-fit")


    #feature = tf.data.Dataset.zip((feature[0],feature[1],feature[2],feature[3],feature[4],feature[5],feature[6],feature[7],feature[8]))
    #num = feature.
    #print(dir(num))

    train_model.fit(
        x,

        steps_per_epoch=len(train_D),
        epochs=15,
        callbacks=[evaluator]
    )