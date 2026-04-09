"""
Обучение и оценка сети Хопфилда.

Обучение: правило Хебба (однократный проход, без итераций).
Оценка:   точность классификации через recall + ближайший паттерн.
"""

import numpy as np
from model import HopfieldNetwork


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def to_bipolar(X: np.ndarray) -> np.ndarray:
    """
    Переводит вещественные значения [0, 1] в биполярные {-1, +1}.
    Порог бинаризации: 0.5.
    """
    return np.where(X >= 0.5, 1.0, -1.0)


def select_prototypes(
    X_bipolar: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    strategy: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Выбирает один «прототип» на класс для запоминания в сети Хопфилда.

    Parameters
    ----------
    X_bipolar : np.ndarray, shape (M, N)
    y         : np.ndarray, shape (M,)
    n_classes : int
    strategy  : 'mean' — усреднённый биполярный вектор класса
                'first' — первый встреченный пример класса

    Returns
    -------
    prototypes : np.ndarray, shape (n_classes, N)
    proto_labels : np.ndarray, shape (n_classes,)
    """
    N = X_bipolar.shape[1]
    prototypes = np.zeros((n_classes, N), dtype=np.float64)
    proto_labels = np.arange(n_classes, dtype=np.int64)

    for c in range(n_classes):
        mask = y == c
        if not np.any(mask):
            continue
        if strategy == "mean":
            mean_vec = X_bipolar[mask].mean(axis=0)
            prototypes[c] = np.where(mean_vec >= 0.0, 1.0, -1.0)
        else:  # 'first'
            prototypes[c] = X_bipolar[mask][0]

    return prototypes, proto_labels


# ---------------------------------------------------------------------------
# Обучение
# ---------------------------------------------------------------------------

def train_model(
    model: HopfieldNetwork,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    proto_strategy: str = "mean",
) -> dict:
    """
    «Обучает» сеть Хопфилда: запоминает прототипы классов.

    Возвращает словарь history с метриками (для совместимости с visualization).
    Поскольку Хопфилд обучается за один проход, история содержит
    единственную точку — точность после запоминания.

    Parameters
    ----------
    model         : HopfieldNetwork
    X_train       : np.ndarray, shape (M, N)  — вещественные [0,1]
    y_train       : np.ndarray, shape (M,)
    X_val         : np.ndarray, shape (V, N)
    y_val         : np.ndarray, shape (V,)
    n_classes     : int
    proto_strategy: 'mean' | 'first'

    Returns
    -------
    history : dict с ключами 'train_acc', 'val_acc'
              (списки длиной 1 — одна «эпоха» обучения Хопфилда)
    """
    X_train_bp = to_bipolar(X_train)
    X_val_bp   = to_bipolar(X_val)

    # Выбираем прототипы и запоминаем их
    prototypes, proto_labels = select_prototypes(
        X_train_bp, y_train, n_classes, strategy=proto_strategy
    )
    model.train(prototypes)

    # Оцениваем
    train_acc = _accuracy(model, X_train_bp, y_train, proto_labels)
    val_acc   = _accuracy(model, X_val_bp,   y_val,   proto_labels)

    history = {
        "train_acc": [train_acc],
        "val_acc":   [val_acc],
        # Для совместимости с visualization (loss не определён для Хопфилда)
        "train_loss": [0.0],
        "val_loss":   [0.0],
        "proto_labels": proto_labels,
    }
    return history


# ---------------------------------------------------------------------------
# Оценка
# ---------------------------------------------------------------------------

def evaluate_model(
    model: HopfieldNetwork,
    X: np.ndarray,
    y: np.ndarray,
    proto_labels: np.ndarray | None = None,
) -> tuple[float, float]:
    """
    Оценивает точность сети Хопфилда.

    Parameters
    ----------
    model        : HopfieldNetwork (уже обученная)
    X            : np.ndarray, shape (M, N)  — вещественные [0,1]
    y            : np.ndarray, shape (M,)
    proto_labels : метки прототипов; если None — используем 0..P-1

    Returns
    -------
    (loss, accuracy) — loss всегда 0.0 (не применимо к Хопфилду)
    """
    X_bp = to_bipolar(X)
    if proto_labels is None:
        proto_labels = np.arange(len(model.patterns), dtype=np.int64)
    acc = _accuracy(model, X_bp, y, proto_labels)
    return 0.0, acc


def _accuracy(
    model: HopfieldNetwork,
    X_bp: np.ndarray,
    y: np.ndarray,
    proto_labels: np.ndarray,
) -> float:
    preds = model.predict(X_bp, proto_labels)
    return float(np.mean(preds == y))
