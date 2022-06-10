import pickle, tfidf
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.models import model_from_json

# テキストを指定（"""で囲むことで改行も改行として認識される）
text1 = """
野球を観るのは楽しいものです。
試合だけでなくインタビューも楽しみです。
"""
text2 = """
常にiPhoneとiPadを持っているので、
二口あるモバイルバッテリがあると便利。
"""
text3 = """
幸せな結婚の秘訣は何でしょうか。
夫には敬意を、妻には愛情を示すことが大切。
"""

# TF-IDFの辞書を読み込む
tfidf.load_dic('text/genre-tdidf.dic')

# Kerasのモデルを定義して重みデータを読み込む
nb_classes = 4
dt_count = len(tfidf.dt_dic)
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(dt_count,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(nb_classes, activation='softmax'))
model.compile(
    loss='categorical_crossentropy',
    optimizer=RMSprop(),
    metrics=['accuracy'])
model.load_weights('./text/genre-model.hdf5')

def check_genre(text):
    """テキストを指定して判定
    """
    LABELS = ['スポーツ', 'IT', '映画', 'ライフ']# ラベルの定義
    data = tfidf.calc_text(text) # TF-IDFのベクトルに変換
    pre = model.predict(np.array([data]))[0] # MLPで予測
    n = pre.argmax()
    print(LABELS[n], '(', pre[n], ')')
    return LABELS[n], float(pre[n]), int(n) 

if __name__ == '__main__':
    check_genre(text1)
    check_genre(text2)
    check_genre(text3)

