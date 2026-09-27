"""Загрузка данных и разбиение train на части для двухэтапной схемы.

Train делится на три непересекающиеся по поисковым сессиям части:
  * val: 3000 сессий для итоговой оценки (те же, что в бейзлайне: тот же seed и порядок сэмплирования);
  * rank: сессии для обучения ранкера (второй этап);
  * stats: всё остальное, по этим строкам считаются приоры (локация, подкатегория).

Зачем отдельная часть под ранкер: если признаки для обучения ранкера посчитать по статистикам,
в которые входят сами эти сессии, признаки будут "подсмотренными" (например, приор подкатегории
для запроса, который уже есть в статистике, будет идеально точным), и ранкер переоценит их.
Сессии сэмплируются равномерно по уникальным текстам запросов, так же, как устроен бенчмарк.
"""
import numpy as np
import pandas as pd

from common import norm_text

KEY = ['q', 'search_location_id', 'search_infm_params_text']
ITEM_COLS = ['item_id', 'item_title_raw', 'item_description_raw', 'item_infm_params_text',
             'item_category_id', 'item_microcat_id', 'item_price', 'item_rating', 'item_rating_reviews_count',
             'item_location_id', 'item_latitude', 'item_longitude', 'item_is_phone_hidden', 'item_is_message_forbidden']


QUERY_COLS = ['search_query', 'search_location_id', 'search_infm_params_text', 'search_category']
MIN_CLEAN_TEXTS = 1000      # меньше этого чистой валидации и обучения ранкера не хватит


def check_inputs(train: pd.DataFrame, bq: pd.DataFrame, bi: pd.DataFrame) -> None:
    """Проверка, что входные файлы имеют ожидаемую структуру (см. README, раздел про входные данные).

    Падает с понятным сообщением, если нет нужных колонок или нарушена уникальность id,
    и предупреждает, если в корпусе мало объявлений из train: на них строится чистая валидация.
    """
    query_cols = QUERY_COLS + ['search_is_delivery_search']          # флаг доставки нужен только для EDA
    required = {'train': (train, query_cols + ITEM_COLS), 'benchmark_queries': (bq, ['query_id'] + query_cols),
                'benchmark_items': (bi, ITEM_COLS)}
    for name, (df, cols) in required.items():
        lost = [c for c in cols if c not in df.columns]
        if lost:
            raise ValueError(f'{name}: нет колонок {lost}')
        if df.empty:
            raise ValueError(f'{name}: файл пустой')
    if bi.item_id.duplicated().any():
        raise ValueError('benchmark_items: item_id должны быть уникальными')
    if bq.query_id.duplicated().any():
        raise ValueError('benchmark_queries: query_id должны быть уникальными')
    share = train.item_id.isin(set(bi.item_id)).mean()
    print(f'Проверка входных данных пройдена. Строк train с объявлением из корпуса: {share:.1%}')
    if share < 0.01:
        print('Внимание: объявлений из train в корпусе почти нет, чистую валидацию (split_in_corpus) не построить.')


def load(data_dir='data'):
    train = pd.read_parquet(f'{data_dir}/train.parquet')
    bq = pd.read_parquet(f'{data_dir}/benchmark_queries.parquet')
    bi = pd.read_parquet(f'{data_dir}/benchmark_items.parquet')[ITEM_COLS]
    train['q'] = train.search_query.map(norm_text)
    bq['q'] = bq.search_query.map(norm_text)
    return train, bq, bi


def sample_sessions(sessions: pd.DataFrame, texts, seed: int) -> pd.DataFrame:
    """По одной случайной сессии на каждый текст из texts."""
    return sessions[sessions.q.isin(set(texts))].sample(frac=1, random_state=seed).drop_duplicates('q')


def to_queries(rows: pd.DataFrame) -> pd.DataFrame:
    """Переводит строки train в одну строку на сессию + множество позитивов (item_ids)."""
    return (rows.groupby(KEY, sort=False)
            .agg(search_query=('search_query', 'first'), search_category=('search_category', 'first'),
                 item_ids=('item_id', set)).reset_index())


def split(train: pd.DataFrame, n_val=3000, n_rank=12000, seed=42):
    """Возвращает (stats_rows, rank_queries, val_queries) и маски строк."""
    rng = np.random.default_rng(seed)
    sessions = train[KEY].drop_duplicates()
    texts = sessions.q.unique()
    val_texts = rng.choice(texts, n_val, replace=False)          # как в бейзлайне
    rest = np.setdiff1d(texts, val_texts)
    rank_texts = np.random.default_rng(seed + 1).choice(rest, n_rank, replace=False)

    val_s = sample_sessions(sessions, val_texts, seed)
    rank_s = sample_sessions(sessions, rank_texts, seed + 1)
    tag = train[KEY].merge(pd.concat([val_s.assign(_p='val'), rank_s.assign(_p='rank')]), on=KEY, how='left')._p.values
    is_val, is_rank = tag == 'val', tag == 'rank'
    return train[~is_val & ~is_rank], to_queries(train[is_rank]), to_queries(train[is_val]), is_val, is_rank


def build_corpus(bi: pd.DataFrame, *row_sets) -> pd.DataFrame:
    """Корпус = benchmark_items + позитивы отложенных сессий, которых в нём нет."""
    extra = pd.concat([r[ITEM_COLS] for r in row_sets]).drop_duplicates('item_id')
    extra = extra[~extra.item_id.isin(set(bi.item_id))]
    return pd.concat([bi, extra], ignore_index=True)


def split_in_corpus(train: pd.DataFrame, bi: pd.DataFrame, n_val=2500, seed=7):
    """Разбиение без артефакта "позитив из train, дистракторы из корпуса".

    В обычном разбиении позитивы отложенных сессий приходится добавлять в корпус, и они
    систематически отличаются от "родных" объявлений benchmark_items (другой период, другие
    статистики). Адверсариальная проверка отличает их с AUC около 0.69, и ранкер может выучить
    этот артефакт вместо релевантности.

    Здесь берём только сессии, где все выбранные объявления есть в benchmark_items
    (~18k сессий, ~11k уникальных текстов). Корпус тогда ровно benchmark_items, как в бенчмарке.
    Тексты делятся на val (n_val, но не больше четверти) и rank (остальные), по одной сессии на текст.
    """
    sessions_ok = (train.assign(ok=train.item_id.isin(set(bi.item_id)))
                   .groupby(KEY).ok.all().loc[lambda s: s].reset_index()[KEY])
    texts = sessions_ok.q.unique()
    if len(texts) < MIN_CLEAN_TEXTS:
        raise ValueError(f'Сессий, где все выбранные объявления есть в корпусе, только {len(texts)} уникальных текстов '
                         f'(нужно хотя бы {MIN_CLEAN_TEXTS}): train и корпус почти не пересекаются.')
    n_val = min(n_val, len(texts) // 4)              # при малом пересечении не больше четверти текстов на val
    rng = np.random.default_rng(seed)
    val_texts = rng.choice(texts, n_val, replace=False)
    rank_texts = np.setdiff1d(texts, val_texts)
    val_s = sample_sessions(sessions_ok, val_texts, seed)
    rank_s = sample_sessions(sessions_ok, rank_texts, seed + 1)
    tag = train[KEY].merge(pd.concat([val_s.assign(_p='val'), rank_s.assign(_p='rank')]), on=KEY, how='left')._p.values
    is_val, is_rank = tag == 'val', tag == 'rank'
    return train[~is_val & ~is_rank], to_queries(train[is_rank]), to_queries(train[is_val])
