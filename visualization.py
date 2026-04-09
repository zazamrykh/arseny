"""
Визуализация результатов экспериментов с сетью Хопфилда.

Графики:
  1. Точность vs порог θ (передаточная функция)
  2. Точность vs Ns (число запоминаемых паттернов / ёмкость)
  3. Матрица ошибок лучшей модели
  4. Текстовый отчёт классификации
  5. Энергетический профиль: энергия recalled-состояний
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # безголовый бэкенд — сохраняем файлы без GUI
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from train import to_bipolar


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def plot_results(results: list[dict], label_map: dict, X_test: np.ndarray, y_test: np.ndarray) -> None:
    _plot_threshold_experiment(results)
    _plot_capacity_experiment(results)
    _plot_confusion_matrix(results, label_map, X_test, y_test)
    _print_classification_report(results, label_map, X_test, y_test)
    _plot_energy_profile(results, X_test)


# ---------------------------------------------------------------------------
# График 1: Влияние порога θ
# ---------------------------------------------------------------------------

def _plot_threshold_experiment(results: list[dict]) -> None:
    exp = [r for r in results if r["experiment"] == "threshold"]
    if not exp:
        return

    thetas = [r["param"] for r in exp]
    accs   = [r["test_acc"] for r in exp]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(thetas, accs, "bo-", linewidth=2, markersize=9)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1, label="θ = 0 (стандарт)")

    for theta, acc in zip(thetas, accs):
        ax.annotate(
            f"{acc:.3f}",
            (theta, acc),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=9,
        )

    ax.set_xlabel("Порог θ передаточной функции нейрона", fontsize=12)
    ax.set_ylabel("Точность на тестовой выборке", fontsize=12)
    ax.set_title(
        "Эксперимент 1: Влияние порога θ на точность\n"
        "Сеть Хопфилда, Ns = все классы",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig("./exp1_threshold.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Сохранён: exp1_threshold.png")


# ---------------------------------------------------------------------------
# График 2: Влияние Ns (ёмкость сети)
# ---------------------------------------------------------------------------

def _plot_capacity_experiment(results: list[dict]) -> None:
    exp = [r for r in results if r["experiment"] == "capacity"]
    if not exp:
        return

    ns_vals = [r["param"] for r in exp]
    accs    = [r["test_acc"] for r in exp]

    # Теоретическая ёмкость Хопфилда (для N=784): 0.138·N ≈ 108
    # Отображаем как вертикальную линию
    N = exp[0]["model"].n_neurons
    theoretical_capacity = int(0.138 * N)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Левый: точность vs Ns ---
    axes[0].plot(ns_vals, accs, "rs-", linewidth=2, markersize=9)
    axes[0].axvline(
        theoretical_capacity,
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label=f"Теор. ёмкость ≈ {theoretical_capacity}",
    )

    for ns, acc in zip(ns_vals, accs):
        axes[0].annotate(
            f"{acc:.3f}",
            (ns, acc),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=9,
        )

    axes[0].set_xlabel("Ns (число запоминаемых паттернов)", fontsize=12)
    axes[0].set_ylabel("Точность на тестовой выборке", fontsize=12)
    axes[0].set_title(
        "Эксперимент 2: Влияние Ns на точность\n"
        "Сеть Хопфилда, θ = 0",
        fontsize=13,
    )
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])

    # --- Правый: точность vs Ns/N (нормированная нагрузка) ---
    loads = [ns / N for ns in ns_vals]
    axes[1].plot(loads, accs, "gs-", linewidth=2, markersize=9)
    axes[1].axvline(
        0.138,
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label="Теор. предел 0.138",
    )
    axes[1].set_xlabel("Нагрузка α = Ns / N", fontsize=12)
    axes[1].set_ylabel("Точность на тестовой выборке", fontsize=12)
    axes[1].set_title(
        "Нормированная нагрузка на сеть",
        fontsize=13,
    )
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig("./exp2_capacity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Сохранён: exp2_capacity.png")


# ---------------------------------------------------------------------------
# График 3: Матрица ошибок лучшей модели
# ---------------------------------------------------------------------------

def _plot_confusion_matrix(
    results: list[dict],
    label_map: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    best = max(results, key=lambda r: r["test_acc"])
    model = best["model"]
    proto_labels = best["proto_labels"]

    X_bp = to_bipolar(X_test)

    # Фильтруем тест по классам, которые модель знает
    known = set(proto_labels.tolist())
    mask  = np.array([yi in known for yi in y_test])
    X_bp_f = X_bp[mask]
    y_f    = y_test[mask]

    preds = model.predict(X_bp_f, proto_labels)

    unique_labels = np.unique(y_f)
    cm = confusion_matrix(y_f, preds, labels=unique_labels)

    n_classes = len(unique_labels)
    fig_size  = max(10, n_classes * 0.45)
    fig, ax   = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    show_numbers = n_classes <= 30
    sns.heatmap(
        cm,
        annot=show_numbers,
        fmt="d",
        cmap="Blues",
        xticklabels=[label_map[i] for i in unique_labels],
        yticklabels=[label_map[i] for i in unique_labels],
        square=True,
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
    )

    ax.set_xlabel("Предсказанная буква", fontsize=12)
    ax.set_ylabel("Истинная буква", fontsize=12)
    ax.set_title(
        f"Матрица ошибок — лучшая модель\n"
        f"{best['experiment']}={best['param']}, "
        f"точность={best['test_acc']:.4f}",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig("./confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Сохранён: confusion_matrix.png")


# ---------------------------------------------------------------------------
# Текстовый отчёт
# ---------------------------------------------------------------------------

def _print_classification_report(
    results: list[dict],
    label_map: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    best = max(results, key=lambda r: r["test_acc"])
    model = best["model"]
    proto_labels = best["proto_labels"]

    X_bp = to_bipolar(X_test)
    known = set(proto_labels.tolist())
    mask  = np.array([yi in known for yi in y_test])
    X_bp_f = X_bp[mask]
    y_f    = y_test[mask]

    preds = model.predict(X_bp_f, proto_labels)

    unique_labels = np.unique(y_f)
    target_names  = [label_map[i] for i in unique_labels]

    print(f"\n{'='*60}")
    print(f"Отчёт классификации — лучшая модель")
    print(f"Параметр: {best['experiment']} = {best['param']}")
    print(f"{'='*60}")
    print(
        classification_report(
            y_f,
            preds,
            labels=unique_labels,
            target_names=target_names,
            zero_division=0,
        )
    )


# ---------------------------------------------------------------------------
# График 4: Энергетический профиль
# ---------------------------------------------------------------------------

def _plot_energy_profile(results: list[dict], X_test: np.ndarray) -> None:
    """
    Для лучшей модели строит гистограмму энергий recalled-состояний.
    Низкая энергия = устойчивый аттрактор = уверенное распознавание.
    """
    best = max(results, key=lambda r: r["test_acc"])
    model = best["model"]

    X_bp = to_bipolar(X_test)
    # Берём не более 200 примеров для скорости
    idx = np.random.choice(len(X_bp), size=min(200, len(X_bp)), replace=False)
    sample = X_bp[idx]

    energies = []
    for x in sample:
        recalled = model.recall(x)
        energies.append(model.energy(recalled))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(energies, bins=30, color="steelblue", edgecolor="black", alpha=0.8)
    ax.set_xlabel("Энергия E = -½ xᵀWx", fontsize=12)
    ax.set_ylabel("Число примеров", fontsize=12)
    ax.set_title(
        "Энергетический профиль recalled-состояний\n"
        f"Лучшая модель: {best['experiment']}={best['param']}",
        fontsize=13,
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("./energy_profile.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Сохранён: energy_profile.png")
