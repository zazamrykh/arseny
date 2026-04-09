"""
Эксперименты с сетью Хопфилда для распознавания рукописных букв.

Изменяемые параметры (согласно заданию):
  1. Передаточная функция нейрона (порог θ)
  2. Ns — число элементов в «скрытом слое» (ёмкость: число запоминаемых паттернов)

Каждый эксперимент возвращает список словарей:
    {
        'experiment' : str   — название эксперимента
        'param'      : any   — значение параметра
        'test_acc'   : float — точность на тесте
        'history'    : dict  — история обучения
        'model'      : HopfieldNetwork
        'proto_labels': np.ndarray
    }
"""

import numpy as np
from model import HopfieldNetwork
from train import train_model, evaluate_model, to_bipolar, select_prototypes


# ---------------------------------------------------------------------------
# Вспомогательная функция: обучить и оценить одну конфигурацию
# ---------------------------------------------------------------------------

def _run_one(
    X_train, y_train,
    X_val,   y_val,
    X_test,  y_test,
    K: int,
    threshold: float,
    Ns: int,
    proto_strategy: str = "mean",
) -> dict:
    """
    Создаёт сеть Хопфилда, обучает её и возвращает результат.

    Parameters
    ----------
    threshold : float
        Порог θ передаточной функции нейрона.
    Ns : int
        Число паттернов, которые сеть запоминает (≤ K).
        Если Ns < K, берём первые Ns классов.
    """
    input_size = X_train.shape[1]

    # Ограничиваем число классов до Ns (эксперимент с ёмкостью)
    if Ns < K:
        mask_train = y_train < Ns
        mask_val   = y_val   < Ns
        mask_test  = y_test  < Ns
        Xt, yt = X_train[mask_train], y_train[mask_train]
        Xv, yv = X_val[mask_val],     y_val[mask_val]
        Xe, ye = X_test[mask_test],   y_test[mask_test]
        n_classes = Ns
    else:
        Xt, yt = X_train, y_train
        Xv, yv = X_val,   y_val
        Xe, ye = X_test,  y_test
        n_classes = K

    model = HopfieldNetwork(
        n_neurons=input_size,
        threshold=threshold,
        max_iter=20,
    )

    history = train_model(
        model, Xt, yt, Xv, yv,
        n_classes=n_classes,
        proto_strategy=proto_strategy,
    )

    proto_labels = history["proto_labels"]
    _, test_acc = evaluate_model(model, Xe, ye, proto_labels)

    return {
        "history":     history,
        "model":       model,
        "proto_labels": proto_labels,
        "test_acc":    test_acc,
    }


# ---------------------------------------------------------------------------
# Главная функция экспериментов
# ---------------------------------------------------------------------------

def run_experiments(X_train, y_train, X_val, y_val, X_test, y_test, K: int) -> list[dict]:
    """
    Запускает два эксперимента и возвращает общий список результатов.
    """
    results = []

    # ===================================================================
    # ЭКСПЕРИМЕНТ 1: Влияние порога θ передаточной функции нейрона
    # Ns зафиксировано = K (все классы)
    # ===================================================================
    print("\n" + "=" * 60)
    print("ЭКСПЕРИМЕНТ 1: Влияние порога θ передаточной функции")
    print("=" * 60)

    thresholds = [-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0]

    for theta in thresholds:
        print(f"\n--- θ = {theta:+.2f} ---")
        res = _run_one(
            X_train, y_train, X_val, y_val, X_test, y_test,
            K=K, threshold=theta, Ns=K,
        )
        print(f"  >>> Точность на тесте: {res['test_acc']:.4f}")
        results.append({
            "experiment":  "threshold",
            "param":       theta,
            "test_acc":    res["test_acc"],
            "history":     res["history"],
            "model":       res["model"],
            "proto_labels": res["proto_labels"],
        })

    # ===================================================================
    # ЭКСПЕРИМЕНТ 2: Влияние Ns — числа запоминаемых паттернов (ёмкость)
    # θ зафиксировано = 0.0 (стандартная функция знака)
    # ===================================================================
    print("\n" + "=" * 60)
    print("ЭКСПЕРИМЕНТ 2: Влияние Ns (число запоминаемых паттернов)")
    print("=" * 60)

    # Теоретическая ёмкость Хопфилда ≈ 0.138·N
    # Для N=784: ~108 паттернов. Мы варьируем от малого до K.
    ns_values = _build_ns_range(K)

    for Ns in ns_values:
        print(f"\n--- Ns = {Ns} паттернов ---")
        res = _run_one(
            X_train, y_train, X_val, y_val, X_test, y_test,
            K=K, threshold=0.0, Ns=Ns,
        )
        print(f"  >>> Точность на тесте: {res['test_acc']:.4f}")
        results.append({
            "experiment":  "capacity",
            "param":       Ns,
            "test_acc":    res["test_acc"],
            "history":     res["history"],
            "model":       res["model"],
            "proto_labels": res["proto_labels"],
        })

    # ===================================================================
    # Итог
    # ===================================================================
    print("\n" + "=" * 60)
    print("ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ")
    best = max(results, key=lambda r: r["test_acc"])
    print(
        f"Лучший результат: {best['experiment']}={best['param']}, "
        f"точность={best['test_acc']:.4f}"
    )
    print("=" * 60)

    return results


# ---------------------------------------------------------------------------
# Вспомогательная: диапазон Ns
# ---------------------------------------------------------------------------

def _build_ns_range(K: int) -> list[int]:
    """
    Строит список значений Ns от 2 до K.
    Если K ≤ 10 — берём все значения 2..K.
    Иначе — логарифмически равномерный набор + K.
    """
    if K <= 10:
        return list(range(2, K + 1))

    # Логарифмически равномерный набор из ~8 точек
    raw = np.unique(
        np.round(np.geomspace(2, K, num=8)).astype(int)
    ).tolist()

    # Гарантируем, что K включён
    if raw[-1] != K:
        raw.append(K)

    return raw
