# from sklearn.datasets import fetch_20newsgroups
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report

# # 加载子集数据
# categories = ['rec.sport.baseball', 'sci.med', 'talk.politics.misc']
# train_data = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
# test_data = fetch_20newsgroups(subset='test', categories=categories, remove=('headers', 'footers', 'quotes'))

# # 文本向量化
# vectorizer = TfidfVectorizer(max_features=500)
# X_train = vectorizer.fit_transform(train_data.data)
# X_test = vectorizer.transform(test_data.data)

# # 分类模型
# clf = LogisticRegression(max_iter=100)
# clf.fit(X_train, train_data.target)
# y_pred = clf.predict(X_test)

# # 评估
# print(classification_report(test_data.target, y_pred, target_names=categories))
import nltk
for resource in ['punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource)

import numpy as np
import gensim.downloader as api
from nltk.tokenize import word_tokenize
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

nltk.download('punkt')

# 加载 GloVe 词向量
w2v = api.load('glove-wiki-gigaword-100')

# 加载数据
categories = ['rec.sport.baseball', 'sci.med', 'talk.politics.misc']
data = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
test = fetch_20newsgroups(subset='test', categories=categories, remove=('headers', 'footers', 'quotes'))

# 文本转向量：平均词向量
def vectorize(texts):
    vectors = []
    for text in texts:
        words = word_tokenize(text.lower())
        word_vecs = [w2v[word] for word in words if word in w2v]
        if word_vecs:
            vectors.append(np.mean(word_vecs, axis=0))
        else:
            vectors.append(np.zeros(100))  # GloVe 是 100维
    return np.array(vectors)

# 转换训练和测试集
X_train = vectorize(data.data)
X_test = vectorize(test.data)

# 训练分类器
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, data.target)
y_pred = clf.predict(X_test)

# 输出分类结果
print("GloVe + Logistic Regression:")
print(classification_report(test.target, y_pred, target_names=categories))
