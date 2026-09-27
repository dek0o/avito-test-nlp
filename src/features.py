"""Признаки пар "запрос - кандидат" для ранкера второго этапа.

Первый этап (BaselineRetriever) отдаёт ~500 кандидатов на запрос, ранкер переупорядочивает их
и берёт топ-50. Признаки делятся на группы:

* первый этап: итоговый скор бейзлайна и ранг по нему;
* текст: BM25 по каждому полю отдельно, доля слов запроса, найденных в заголовке и во всём тексте;
* локация: лог-приор p(item_loc | search_loc), совпадение id, расстояние в км до "центра" локации поиска,
  плотность объявлений в локации;
* подкатегория: p(microcat | q) из kNN, её ранг среди подкатегорий запроса;
* фильтры: доля фраз "Вид/Тип услуги" запроса, найденных в параметрах;
* объявление: рейтинг, отзывы, цена, формат работы (выезд / удалённо), длина текстов, флаги;
* запрос: длина, был ли такой текст в train, сходство с ближайшим запросом train;
* dense: косинус эмбеддингов запроса и объявления (добавляется, если переданы эмбеддинги).
"""
import numpy as np
import pandas as pd

from common import haversine, norm_text, parse_filters, to_float


class FeatureBuilder:
    def __init__(self, retriever, stats: pd.DataFrame, item_emb=None):
        """retriever: обученный BaselineRetriever (его корпус совпадает с корпусом поиска), stats: строки train для статистик."""
        self.r = retriever
        items = retriever.items
        self.item_emb = item_emb
        self.stats_q = set(stats.q)

        # статические признаки объявлений (не зависят от запроса)
        params = items.item_infm_params_text.fillna('')
        title_norm = items.item_title_raw.map(norm_text)
        self.item_feats = pd.DataFrame({
            'rating': items.item_rating.astype(float),
            'reviews_log': np.log1p(items.item_rating_reviews_count.fillna(0).astype(float)),
            'price_log': np.log1p(to_float(items.item_price).clip(lower=0)),
            'phone_hidden': items.item_is_phone_hidden.astype(np.int8),
            'msg_forbidden': items.item_is_message_forbidden.astype(np.int8),
            'remote': params.str.contains('Удалённо|Удаленно|онлайн', case=False, regex=True).astype(np.int8),
            'visit': params.str.contains('Выезд|выезжаете|у клиента', case=False, regex=True).astype(np.int8),
            'title_len': title_norm.str.split().str.len().fillna(0),
            'desc_len_log': np.log1p(items.item_description_raw.fillna('').str.len()),
            'title_dups_log': np.log1p(title_norm.map(title_norm.value_counts())),
        }).astype(np.float32).values
        self.item_feat_names = ['rating', 'reviews_log', 'price_log', 'phone_hidden', 'msg_forbidden', 'remote',
                                'visit', 'title_len', 'desc_len_log', 'title_dups_log']
        # Популярность объявления в train не используем: в корпусе бенчмарка лишь ~10% объявлений встречались
        # в train, а на валидации все правильные ответы по построению из train, и признак переоценивался бы.

        # координаты объявлений и "центры" локаций поиска
        self.lat = to_float(items.item_latitude).values.astype(np.float32)
        self.lon = to_float(items.item_longitude).values.astype(np.float32)
        s = stats.assign(lat=to_float(stats.item_latitude), lon=to_float(stats.item_longitude))
        c_search = s.groupby('search_location_id')[['lat', 'lon']].median()          # куда реально ходят из локации
        c_item = pd.DataFrame({'loc': items.item_location_id, 'lat': self.lat, 'lon': self.lon}
                              ).groupby('loc')[['lat', 'lon']].median()                  # центр по объявлениям
        self.centroid = pd.concat([c_search, c_item[~c_item.index.isin(c_search.index)]])
        self.loc_density = items.item_location_id.value_counts()
        self.item_loc = items.item_location_id.values
        # сколько в корпусе объявлений той же подкатегории в той же локации: мера конкуренции за место в топ-50
        mc_loc = items.item_microcat_id.astype(str) + '_' + items.item_location_id.astype(str)
        self.mc_loc_density_log = np.log1p(mc_loc.map(mc_loc.value_counts()).values).astype(np.float32)

        # символьные n-граммы заголовков: сходство с запросом, устойчивое к опечаткам ("ремон генераторов")
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.title_norm = title_norm.values
        self.char_tv = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), sublinear_tf=True, min_df=3,
                                       dtype=np.float32)
        self.title_char = self.char_tv.fit_transform(title_norm)

        # бинарные матрицы "терм есть в поле" для доли покрытия слов запроса
        tfs = retriever.bm25.tfs
        self.title_bin = (tfs['title'] > 0).astype(np.float32).T.tocsr()
        allf = tfs['title'] + tfs['params'] + tfs['desc']
        self.all_bin = (allf > 0).astype(np.float32).T.tocsr()

    def build(self, queries: pd.DataFrame, cand: np.ndarray, s1: np.ndarray, q_emb=None, batch=256) -> pd.DataFrame:
        """queries: n строк; cand/s1: матрицы (n x k), индексы кандидатов в корпусе и скоры первого этапа.

        Возвращает длинную таблицу n*k строк: qid, doc, признаки.
        """
        r = self.r
        n, k = cand.shape
        out = []
        for s in range(0, n, batch):
            qb = queries.iloc[s:s + batch]
            cb, sb = cand[s:s + batch], s1[s:s + batch]
            b = len(qb)
            Q = r.bm25.query_matrix(qb.search_query.tolist())
            nq = np.asarray(Q.sum(axis=1)).ravel().clip(min=1)
            take = lambda M: np.take_along_axis(np.asarray((Q @ M).todense(), dtype=np.float32), cb, axis=1)
            f = {}
            f['s1_score'] = sb
            f['s1_rank'] = np.argsort(np.argsort(-sb, axis=1), axis=1).astype(np.float32)
            for fld, M in r.bm25.WT_fields.items():
                f[f'bm25_{fld}'] = take(M)
            f['bm25_sum'] = f['bm25_title'] + f['bm25_params'] + f['bm25_desc']
            f['bm25_rel'] = f['bm25_sum'] / (f['bm25_sum'].max(axis=1, keepdims=True) + 1e-9)
            f['cover_title'] = take(self.title_bin) / nq[:, None]
            qn = qb.search_query.map(norm_text).values
            qc = self.char_tv.transform(qn)
            f['title_char_cos'] = np.vstack([(self.title_char[cb[i]] @ qc[i].T).toarray().ravel() for i in range(b)])
            f['title_phrase'] = np.array([[q in t for t in self.title_norm[cb[i]]] for i, q in enumerate(qn)],
                                         dtype=np.float32)
            f['cover_all'] = take(self.all_bin) / nq[:, None]

            # подкатегория
            P = r.mc.predict(qb.search_query.map(norm_text))
            mc_rank = np.argsort(np.argsort(-P, axis=1), axis=1)
            dmc = r.doc_mc[cb]
            f['mc_p'] = np.take_along_axis(P, dmc, axis=1)
            f['mc_rank'] = np.take_along_axis(mc_rank, dmc, axis=1).astype(np.float32)
            # максимальное сходство с запросом train: насколько запрос "знаком"
            sim = (r.mc.tv.transform(qb.search_query.map(norm_text)) @ r.mc.Xt)
            qmaxsim = np.asarray(sim.max(axis=1).todense()).ravel()

            # локация, фильтры, расстояние
            loc_lp = np.empty((b, k), np.float32)
            flt = np.zeros((b, k), np.float32)
            dist = np.empty((b, k), np.float32)
            for i, (sl, ft) in enumerate(zip(qb.search_location_id, qb.search_infm_params_text)):
                c = cb[i]
                table = r.loc.table.get(sl, {sl: 1.0})
                loc_lp[i] = np.log(np.array([table.get(x, 0.0) for x in self.item_loc[c]], np.float32) + 1e-4)
                ph = parse_filters(ft)
                if ph:
                    flt[i] = np.mean([r._filter_hits(p)[c] for p in ph], axis=0)
                if sl in self.centroid.index:
                    clat, clon = self.centroid.loc[sl]
                    dist[i] = haversine(clat, clon, self.lat[c], self.lon[c])
                else:
                    dist[i] = np.nan
            f['loc_logp'] = loc_lp
            f['same_loc'] = (self.item_loc[cb] == qb.search_location_id.values[:, None]).astype(np.float32)
            f['dist_log'] = np.log1p(dist)
            f['dist_rank'] = np.argsort(np.argsort(np.nan_to_num(dist, nan=1e9), axis=1), axis=1).astype(np.float32)
            f['mc_loc_density_log'] = self.mc_loc_density_log[cb]
            f['filter_frac'] = flt

            if self.item_emb is not None and q_emb is not None:
                qe = q_emb[s:s + batch].astype(np.float32)
                f['dense_cos'] = np.einsum('bd,bkd->bk', qe, self.item_emb[cb].astype(np.float32))
                f['dense_rank'] = np.argsort(np.argsort(-f['dense_cos'], axis=1), axis=1).astype(np.float32)

            # признаки запроса (повторяются на всех кандидатах)
            qf = {
                'q_words': qb.q.str.split().str.len().values,
                'q_seen': qb.q.isin(self.stats_q).values,
                'q_maxsim': qmaxsim,
                'q_n_filters': qb.search_infm_params_text.map(lambda x: len(parse_filters(x))).values,
                'q_region': ~qb.search_location_id.isin(self.loc_density.index).values,
                'q_loc_density_log': np.log1p(qb.search_location_id.map(self.loc_density).fillna(0).values),
                'q_cat0': (qb.search_category.values == 0) if 'search_category' in qb else np.zeros(b, bool),
            }
            block = {name: v.ravel() for name, v in f.items()}
            for name, v in qf.items():
                block[name] = np.repeat(np.asarray(v, dtype=np.float32), k)
            itf = self.item_feats[cb.ravel()]
            for j, name in enumerate(self.item_feat_names):
                block[name] = itf[:, j]
            block['qid'] = np.repeat(np.arange(s, s + b), k)
            block['doc'] = cb.ravel()
            out.append(pd.DataFrame(block))
        return pd.concat(out, ignore_index=True)
