"""Общие утилиты: нормализация и токенизация текста, BM25-индекс, метрика, геометрия.

Используется и в ноутбуке, и в скриптах, чтобы логика не дублировалась.
Open-source зависимости: numpy, pandas, scipy, scikit-learn, PyStemmer (Snowball-стеммер).
"""
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp
import Stemmer
from sklearn.feature_extraction.text import CountVectorizer

_stemmer = Stemmer.Stemmer('russian')
_stem_cache: dict[str, str] = {}
_TOKEN_RE = re.compile(r'[a-zа-я0-9]+')


def norm_text(s) -> str:
    """Нижний регистр, ё в е, пунктуация в пробел."""
    s = str(s or '').lower().replace('ё', 'е')
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', s)).strip()


def tokenize(s) -> list[str]:
    """Токены переводятся в стеммы Snowball. Кеш по словам: уникальных слов на порядки меньше, чем вхождений."""
    out = []
    for w in _TOKEN_RE.findall(str(s or '').lower().replace('ё', 'е')):
        st = _stem_cache.get(w)
        if st is None:
            st = _stem_cache[w] = _stemmer.stemWord(w)
        out.append(st)
    return out


def to_float(col: pd.Series) -> pd.Series:
    """Цены и координаты лежат как Decimal/строки: переводим в float."""
    return pd.to_numeric(col.astype(str), errors='coerce')


# BM25

class BM25Index:
    """Многополевой BM25 (сумма BM25 по полям с весами) поверх разреженных матриц.

    Идея: для каждого поля считаем матрицу весов w[d, t] = idf(t) * tf*(k1+1) / (tf + k1*(1-b+b*dl/avgdl)),
    складываем поля с весами и получаем одну матрицу W (docs x vocab).
    Тогда скор запроса равен сумме столбцов W по термам запроса: W @ q.
    IDF общий для документа целиком (терм встречается хоть в одном поле).
    """

    def __init__(self, field_weights: dict[str, float], k1: float = 1.2, b: float = 0.75, b_per_field=None):
        self.field_weights = field_weights
        self.k1, self.b = k1, b
        self.b_per_field = b_per_field or {}

    def fit(self, docs: pd.DataFrame):
        """Токенизация полей + построение весов."""
        self.vec = CountVectorizer(analyzer=tokenize, dtype=np.float32)
        self.vec.fit(pd.concat([docs[f] for f in self.field_weights], ignore_index=True))
        tfs = {f: self.vec.transform(docs[f]).tocsr() for f in self.field_weights}
        return self.fit_tfs(self.vec, tfs)

    def fit_tfs(self, vec, tfs: dict):
        """Построение BM25-весов по уже посчитанным матрицам частот (удобно для перебора весов полей)."""
        self.vec = vec
        n = next(iter(tfs.values())).shape[0]
        present = None
        for f in self.field_weights:
            p = (tfs[f] > 0).astype(np.float32)
            present = p if present is None else present + p
        df = np.asarray((present > 0).sum(axis=0)).ravel()
        idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        W = None
        self.tfs = tfs
        self.WT_fields = {}         # BM25 по каждому полю отдельно (без веса поля), это признаки для ранкера
        for f, wf in self.field_weights.items():
            tf = tfs[f]
            b = self.b_per_field.get(f, self.b)
            dl = np.asarray(tf.sum(axis=1)).ravel()
            norm = self.k1 * (1 - b + b * dl / max(dl.mean(), 1e-9))
            tf = tf.tocoo()
            data = tf.data * (self.k1 + 1) / (tf.data + norm[tf.row]) * idf[tf.col] * wf
            Wf = sp.csr_matrix((data.astype(np.float32), (tf.row, tf.col)), shape=tf.shape)
            self.WT_fields[f] = Wf.T.tocsr() / wf
            W = Wf if W is None else W + Wf
        self.idf = idf
        self.WT = W.T.tocsr()       # vocab x docs: скоры батча запросов = Q @ WT
        return self

    def query_matrix(self, queries) -> sp.csr_matrix:
        """Бинарный вектор термов запроса (повтор слова не усиливает вес)."""
        Q = self.vec.transform(queries)
        Q.data[:] = 1
        return Q

    def score(self, Q: sp.csr_matrix) -> np.ndarray:
        """Плотная матрица скоров (batch x docs)."""
        return np.asarray((Q @ self.WT).todense(), dtype=np.float32)


# метрика

def recall_at_k(pred: list, gold: list, k: int = 50) -> np.ndarray:
    """Recall@k по каждому запросу: |top-k ∩ релевантные| / |релевантные|."""
    return np.array([len(set(p[:k]) & g) / len(g) for p, g in zip(pred, gold)])


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


# фильтры запроса

_FILTER_SPLIT_RE = re.compile(r'(Тип услуги автосервиса|Вид услуги|Тип услуги|Онлайн-запись|Рейтинг пользователя)')
_FILTER_KEYS = {'Тип услуги автосервиса', 'Вид услуги', 'Тип услуги'}


def parse_filters(s) -> list[str]:
    """Достаёт из текста фильтров пары "ключ значение", например 'Вид услуги Красота, здоровье'.

    Текст режется по известным ключам; пустые значения ('Вид услуги' без значения) пропускаем,
    потому что они ничего не ограничивают. Онлайн-запись/рейтинг в параметрах объявления не отражены, их тоже пропускаем.
    Фразы возвращаются в том виде, в котором встречаются в параметрах объявления.
    """
    parts = _FILTER_SPLIT_RE.split(str(s or ''))
    out = []
    for key, val in zip(parts[1::2], parts[2::2]):
        val = val.strip()
        if key in _FILTER_KEYS and val:
            out.append(f'{key} {val}')
    return out
