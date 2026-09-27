"""Dense-ретривер: эмбеддинги запросов и объявлений моделью deepvk/USER2-base (open-source, локально).

USER2 обучена с префиксами задач: запросы кодируются с 'search_query: ', документы с 'search_document: '.
Текст объявления: заголовок + начало описания, обрезка до 128 токенов. Заголовок несёт суть услуги,
а начало описания добавляет синонимы и детали. 256 токенов работают вдвое медленнее без ощутимой пользы.

Размерность. USER2 обучена с Matryoshka (MRL): первые k координат эмбеддинга сами по себе являются
осмысленным эмбеддингом. Храним первые 128 из 768 координат (после обрезки нормируем заново).
На валидации это не меняет полноту этапа 1 (0.987 @500 и при 768, и при 128), а файл эмбеддингов
корпуса уменьшается с 290 до 48 МБ и помещается в git-репозиторий.

Готовые эмбеддинги лежат в artifacts/ и сопоставляются по item_id (для запросов по тексту).
Если корпус или запросы изменились, кодируются только новые объявления и запросы, а файл
в artifacts/ перезаписывается под текущие данные. Полное кодирование корпуса (~190k объявлений)
на Apple M4 (MPS) занимает около 1.5 часов, поэтому при большом числе новых объявлений промежуточный
результат кешируется по чанкам в cache/, и прерванный прогон можно продолжить.
"""
import os

import numpy as np
import pandas as pd

MODEL_NAME = 'deepvk/USER2-base'
EMB_DIM = 128


def item_text(items: pd.DataFrame) -> list[str]:
    return (items.item_title_raw.fillna('') + '\n' + items.item_description_raw.fillna('').str[:600]).tolist()


def truncate(X: np.ndarray, dim: int = EMB_DIM) -> np.ndarray:
    """Matryoshka-обрезка: первые dim координат + L2-нормировка."""
    X = X[:, :dim].astype(np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-9)


def load_model(device=None, max_len=128):
    import torch
    from sentence_transformers import SentenceTransformer
    if device is None:              # GPU NVIDIA, затем Apple MPS, иначе CPU
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    m = SentenceTransformer(MODEL_NAME, device=device)
    m.max_seq_length = max_len
    return m


def _encode_items_chunked(items: pd.DataFrame, cache_prefix: str, chunk=8192, batch_size=64) -> np.ndarray:
    """Полноразмерные эмбеддинги объявлений с покусочным кешем cache_prefix_<i>.npy."""
    texts = item_text(items)
    model, parts = None, []
    for i, s in enumerate(range(0, len(texts), chunk)):
        fn = f'{cache_prefix}_{i:03d}.npy'
        expected = len(texts[s:s + chunk])
        if os.path.exists(fn) and len(np.load(fn, mmap_mode='r')) != expected:
            os.remove(fn)                     # кеш от другого корпуса: пересчитываем
        if not os.path.exists(fn):
            model = model or load_model()
            e = model.encode(texts[s:s + chunk], batch_size=batch_size, prompt_name='search_document',
                             normalize_embeddings=True, convert_to_numpy=True)
            np.save(fn, e.astype(np.float16))
            print(f'  chunk {i}: {s + len(e)}/{len(texts)}', flush=True)
        parts.append(np.load(fn))
    return np.vstack(parts)


def _lookup(keys: np.ndarray, path: str, key_name: str):
    """Позиции keys в сохранённом артефакте (-1, если ключа нет) и сами эмбеддинги артефакта."""
    if not os.path.exists(path):
        return np.full(len(keys), -1), None
    z = np.load(path, allow_pickle=False)
    return pd.Index(z[key_name]).get_indexer(keys), z['emb']


def item_embeddings(items: pd.DataFrame, path='artifacts/emb_items_user2_128.npz',
                    cache_prefix='cache/emb/user2_bi') -> np.ndarray:
    """Эмбеддинги корпуса (n x EMB_DIM, float32) в порядке строк items.

    Берутся из artifacts/ по item_id. Объявления, которых там нет, кодируются моделью.
    """
    ids = np.array(items.item_id.astype(str).tolist(), dtype='S')     # id в байтах: файл < 50 МБ
    pos, saved = _lookup(ids, path, 'item_id')
    missing = pos < 0
    if not missing.any():
        return saved[pos].astype(np.float32)
    print(f'кодируем {missing.sum()} объявлений, которых нет в {path}', flush=True)
    os.makedirs(os.path.dirname(cache_prefix), exist_ok=True)
    prefix = cache_prefix if missing.all() else f'{cache_prefix}_new{missing.sum()}'
    emb = np.empty((len(items), EMB_DIM), np.float32)
    emb[missing] = truncate(_encode_items_chunked(items[missing], prefix))
    if saved is not None:
        emb[~missing] = saved[pos[~missing]]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, item_id=ids, emb=emb.astype(np.float16))
    return emb


def query_embeddings(queries: pd.Series, path: str, batch_size=256) -> np.ndarray:
    """Эмбеддинги запросов (n x EMB_DIM, float32). Берутся из файла по тексту, новые кодируются моделью."""
    texts = np.array(queries.astype(str).tolist(), dtype=str)   # строковый массив, не object: читается без pickle
    pos, saved = _lookup(texts, path, 'text')
    missing = pos < 0
    if not missing.any():
        return saved[pos].astype(np.float32)
    emb = np.empty((len(texts), EMB_DIM), np.float32)
    new = load_model().encode(list(texts[missing]), batch_size=batch_size, prompt_name='search_query',
                              normalize_embeddings=True, convert_to_numpy=True)
    emb[missing] = truncate(new)
    if saved is not None:
        emb[~missing] = saved[pos[~missing]]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, text=texts, emb=emb.astype(np.float16))
    return emb
