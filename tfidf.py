import MeCab
import pickle
import numpy as np

tagger = MeCab.Tagger('-d /opt/homebrew/lib/mecab/dic/mecab-ipadic-neologd') # MeCabの初期化

# TF-IDFとは
# 文書内の単語の出現頻度に、その単語の重要度（全文書における単語の出現頻度の対数）を掛け合わせたもの
# TF_IDF(t) : TF_IDF(t) = tf(t,d) * idf(t) / tf(t,d):文書内における単語の出現頻度 / idf(t):全文書における単語の出現頻度
# idf(t) : idf(t) = log{|D| / df(d,t)} / D:文書の総数 / df(d,t):単語tを含む文書数

word_dic = {'_id': 0} # 単語辞書
dt_dic = {} # 文書全体での単語の出現回数
files = [] # 全文書をIDで保存

def tokenize(text):
    """MeCabで形態素解析を行う
    """
    result = []
    word_s = tagger.parse(text)
    for n in word_s.split('\n'):
        if n == 'EOS' or n == '': continue
        p = n.split('\t')[1].split(',')
        h, h2, org = (p[0], p[1], p[6])
        if not (h in ['名詞', '動詞', '形容詞']): continue
        if h == '名詞' and h2 == '数': continue
        result.append(org)
    return result

def words_to_ids(words, auto_add = True):
    """単語一覧をIDの一覧に変換する
    """
    result = []
    for w in words:
        if w in word_dic:
            result.append(word_dic[w])
            continue
        elif auto_add:
            id = word_dic[w] = word_dic['_id']
            word_dic['_id'] += 1
            result.append(id)
    return result

def add_text(text):
    """テキストをIDリストに変換して追加
    """
    ids = words_to_ids(tokenize(text))
    files.append(ids)

def add_file(path):
    """テキストファイルを学習用に追加する
    """
    with open(path, 'r', encoding='utf-8') as f:
        s = f.read()
        add_text(s)

def calc_files():
    """追加したファイルを計算
    """
    global dt_dic
    result = []
    doc_count = len(files)
    dt_dic = {}
    # 単語の出現頻度を数える
    for words in files:
        used_word = {}
        data = np.zeros(word_dic['_id'])
        for id in words:
            data[id] += 1
            used_word[id] = 1
        # 単語tが使われていればdt_dicを加算
        for id in used_word:
            if not(id in dt_dic): dt_dic[id] = 0
            dt_dic[id] += 1
        # 出現回数を割合に直す
        data = data / len(words) 
        result.append(data)
    # TF-IDFを計算
    for i, doc in enumerate(result):
        for id, v in enumerate(doc):
            idf = np.log(doc_count / dt_dic[id]) + 1
            doc[id] = min([doc[id] * idf, 1.0])
        result[i] = doc
    return result

def save_dic(fname):
    """辞書をファイルへ保存する
    """
    pickle.dump(
        [word_dic, dt_dic, files],
        open(fname, 'wb'))

def load_dic(fname):
    """辞書をファイルから読み込む
    """
    global word_dic, dt_dic, files
    n = pickle.load(open(fname, 'rb'))
    word_dic, dt_dic, files = n

def calc_text(text):
    """辞書を更新せずにベクトル変換する
    """
    data = np.zeros(word_dic['_id'])
    words = words_to_ids(tokenize(text), False)
    for w in words:
        data[w] += 1
    data = data / len(words)
    for id, v in enumerate(data):
        idf = np.log(len(files) / dt_dic[id]) + 1
        data[id] = min([data[id] * idf, 1.0])
    return data

# モジュールのテスト
if __name__ == '__main__':
    add_text('雨')
    add_text('今日は、雨が降った。')
    add_text('今日は暑い日だったけど雨が降った。')
    add_text('今日も雨だ。でも日曜だ。')
    print(calc_files())
    print(word_dic)


