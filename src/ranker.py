"""Второй этап: LightGBM LambdaRank поверх ~500 кандидатов первого этапа.

LambdaRank оптимизирует качество верха списка внутри каждого запроса (группы). Отсечение
lambdarank_truncation_level=60 фокусирует обучение на границе топ-50, которая и важна для Recall@50.
Для обучения оставляем только запросы, у которых позитив попал в пул: у остальных ранкеру
нечему учиться.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

NON_FEATURES = ['qid', 'doc', 'label']
LGB_PARAMS = dict(objective='lambdarank', n_estimators=600, learning_rate=0.05, num_leaves=63,
                  min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  lambdarank_truncation_level=60, verbose=-1, n_jobs=8, random_state=0)


def feature_names(F: pd.DataFrame) -> list[str]:
    return [c for c in F.columns if c not in NON_FEATURES]


def fit_ranker(F: pd.DataFrame, feats=None) -> lgb.LGBMRanker:
    feats = feats or feature_names(F)
    F = F[F.groupby('qid').label.transform('max') > 0]          # запросы без позитива в пуле не учат
    m = lgb.LGBMRanker(**LGB_PARAMS)
    m.fit(F[feats], F.label, group=F.groupby('qid', sort=False).size().values)
    return m


def top_k(F: pd.DataFrame, score: np.ndarray, k: int = 50) -> pd.DataFrame:
    """Топ-k кандидатов на запрос по скору ранкера (qid, doc)."""
    return (F[['qid', 'doc']].assign(s=score)
            .sort_values(['qid', 's'], ascending=[True, False]).groupby('qid').head(k))


def recall_from_features(F: pd.DataFrame, score: np.ndarray, n_pos: np.ndarray, k: int = 50) -> np.ndarray:
    """Recall@k на запрос. Позитивы, не попавшие в пул, считаются пропущенными (n_pos: число всех позитивов)."""
    top = F[['qid', 'label']].assign(s=score).sort_values(['qid', 's'], ascending=[True, False]).groupby('qid').head(k)
    hits = top.groupby('qid').label.sum().reindex(range(len(n_pos)), fill_value=0).values
    return hits / n_pos
