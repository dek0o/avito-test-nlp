"""Бейзлайн кандидатогенерации: BM25 + приор локации + приор подкатегории + фильтры запроса.

Итоговый скор объявления d для запроса q является линейной комбинацией четырёх компонент:

    score(q, d) = bm25(q, d) / max_d' bm25(q, d')              # текстовая релевантность, нормирована в [0, 1]
                + a * log(p(loc_d | search_loc_q) + 1e-4)       # насколько типична локация объявления для локации поиска
                + c * log(p(microcat_d | q) + 1e-3)             # насколько типична подкатегория для такого запроса
                + f * (доля фильтров запроса, найденных в параметрах d − 1)   # штраф за несовпадение фильтров

и в ответ идут 50 объявлений с наибольшим скором.

Все статистики (p(loc | loc), p(microcat | q)) считаются только по обучающей части train,
поэтому на валидации нет утечки: отложенные поисковые сессии в них не участвуют.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from common import BM25Index, norm_text, parse_filters


@dataclass
class Params:
    field_weights: tuple = (('title', 3.0), ('params', 1.0), ('desc', 2.0))
    k1: float = 1.2
    b: float = 0.75
    a_loc: float = 0.1       # вес лог-приора локации
    c_mc: float = 0.03       # вес лог-приора подкатегории
    f_filter: float = 0.3    # вес штрафа за несовпадение фильтров
    knn_k: int = 30          # сколько похожих запросов train брать для приора подкатегории
    knn_pow: float = 4.0     # степень сходства: чем больше, тем сильнее доминируют самые похожие
    top_k: int = 50
    batch: int = 256


def item_docs(items: pd.DataFrame) -> pd.DataFrame:
    """Текстовые поля объявления, которые индексирует BM25."""
    return pd.DataFrame({
        'title': items.item_title_raw.fillna(''),
        'params': items.item_infm_params_text.fillna(''),
        'desc': items.item_description_raw.fillna(''),
    })


class LocationPrior:
    """p(item_location | search_location) по train.

    Нужен, потому что локации иерархичны: поиск по "Москва и область" (107620) приводит
    к объявлениям в Москве (637640) и городах области. Для локации поиска, которой нет в train,
    считаем, что объявления берутся из этой же локации.
    """

    def fit(self, train: pd.DataFrame):
        cnt = train.groupby(['search_location_id', 'item_location_id']).size().rename('n').reset_index()
        cnt['p'] = cnt.n / cnt.groupby('search_location_id').n.transform('sum')
        self.table = {sl: dict(zip(g.item_location_id, g.p)) for sl, g in cnt.groupby('search_location_id')}
        return self

    def doc_logp(self, search_loc: int, item_locs: pd.Series, eps: float = 1e-4) -> np.ndarray:
        dist = self.table.get(search_loc, {search_loc: 1.0})
        return np.log(item_locs.map(dist).fillna(0).values.astype(np.float32) + eps)


class MicrocatPrior:
    """p(microcat | запрос) через kNN по текстам запросов train.

    Для нового запроса ищем K самых похожих запросов train (TF-IDF по символьным n-граммам,
    устойчиво к опечаткам и словоформам) и усредняем их распределения подкатегорий
    с весами sim^pow. Если такой же запрос был в train, его сходство = 1 и он доминирует.
    """

    def __init__(self, k: int = 30, pw: float = 4.0):
        self.k, self.pw = k, pw

    def fit(self, train: pd.DataFrame, microcats: pd.Series):
        qm = train.groupby(['q', 'item_microcat_id']).size().rename('n').reset_index()
        self.uq = qm.q.unique()
        qpos = pd.Series(np.arange(len(self.uq)), index=self.uq)
        self.mc_index = pd.Index(pd.concat([microcats, qm.item_microcat_id]).unique())
        M = sp.csr_matrix((qm.n.values.astype(np.float32),
                           (qpos[qm.q].values, self.mc_index.get_indexer(qm.item_microcat_id))),
                          shape=(len(self.uq), len(self.mc_index)))
        self.M = sp.csr_matrix(M.multiply(1 / M.sum(axis=1)))       # строки являются распределениями
        self.tv = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), sublinear_tf=True, min_df=2)
        self.Xt = self.tv.fit_transform(self.uq).T.tocsr()
        return self

    def predict(self, queries: pd.Series) -> np.ndarray:
        """Матрица (n_queries x n_microcats) вероятностей."""
        sim = (self.tv.transform(queries) @ self.Xt).toarray()
        top = np.argpartition(-sim, self.k, axis=1)[:, :self.k]
        P = np.vstack([self.M[top[i]].T @ (sim[i, top[i]] ** self.pw) for i in range(len(queries))])
        P = np.asarray(P, dtype=np.float32)
        return P / (P.sum(axis=1, keepdims=True) + 1e-9)


class BaselineRetriever:
    def __init__(self, params: Params = Params()):
        self.p = params

    def fit(self, train: pd.DataFrame, items: pd.DataFrame, tfs=None):
        """train: пары "запрос - объявление" (с колонкой q), items: корпус, в котором ищем.

        tfs: опционально заранее посчитанные (vectorizer, матрицы частот) корпуса: токенизация
        описаний является самым долгим шагом (~40 с), при переборе параметров её удобно кешировать.
        """
        p = self.p
        self.items = items.reset_index(drop=True)
        self.bm25 = BM25Index(dict(p.field_weights), p.k1, p.b)
        if tfs is None:
            self.bm25.fit(item_docs(self.items))
        else:
            self.bm25.fit_tfs(*tfs)
        self.loc = LocationPrior().fit(train)
        self.mc = MicrocatPrior(p.knn_k, p.knn_pow).fit(train, self.items.item_microcat_id)
        self.doc_mc = self.mc.mc_index.get_indexer(self.items.item_microcat_id)
        self._filter_cache = {}
        return self

    def _filter_hits(self, phrase: str) -> np.ndarray:
        """Булев вектор: есть ли фраза фильтра в параметрах объявления (кешируется по фразе)."""
        if phrase not in self._filter_cache:
            self._filter_cache[phrase] = (self.items.item_infm_params_text.fillna('')
                                          .str.contains(phrase, regex=False).values)
        return self._filter_cache[phrase]

    def score_batch(self, qb: pd.DataFrame) -> np.ndarray:
        p = self.p
        S = self.bm25.score(self.bm25.query_matrix(qb.search_query.tolist()))
        S /= S.max(axis=1, keepdims=True) + 1e-9
        P = self.mc.predict(qb.search_query.map(norm_text))
        S += p.c_mc * np.log(P[:, self.doc_mc] + 1e-3)
        for i, (sl, flt) in enumerate(zip(qb.search_location_id, qb.search_infm_params_text)):
            S[i] += p.a_loc * self.loc.doc_logp(sl, self.items.item_location_id)
            phrases = parse_filters(flt)
            if phrases and p.f_filter:
                S[i] += p.f_filter * (np.mean([self._filter_hits(ph) for ph in phrases], axis=0) - 1)
        return S

    def candidates(self, queries: pd.DataFrame, n: int, q_emb=None, item_emb=None, dense_w: float = 0.0
                   ) -> tuple[np.ndarray, np.ndarray]:
        """Первый этап двухэтапной схемы: индексы (в self.items) и скоры top-n объявлений на запрос.

        Если переданы эмбеддинги, к скору бейзлайна добавляется dense_w * cos(запрос, объявление):
        так в пул попадают объявления, близкие по смыслу, но без совпадения слов.
        """
        idx, sc = [], []
        for s in range(0, len(queries), self.p.batch):
            S = self.score_batch(queries.iloc[s:s + self.p.batch])
            if dense_w and q_emb is not None:
                S += dense_w * (q_emb[s:s + self.p.batch].astype(np.float32) @ item_emb.T)
            top = np.argpartition(-S, n, axis=1)[:, :n]
            idx.append(top)
            sc.append(np.take_along_axis(S, top, axis=1))
        return np.vstack(idx), np.vstack(sc)

    def predict(self, queries: pd.DataFrame) -> list[list[str]]:
        """Для каждого запроса возвращает список из top_k item_id (порядок по убыванию скора)."""
        ids = self.items.item_id.values
        out = []
        for s in range(0, len(queries), self.p.batch):
            S = self.score_batch(queries.iloc[s:s + self.p.batch])
            top = np.argpartition(-S, self.p.top_k, axis=1)[:, :self.p.top_k]
            order = np.take_along_axis(S, top, axis=1).argsort(axis=1)[:, ::-1]
            out += [list(ids[r]) for r in np.take_along_axis(top, order, axis=1)]
        return out
