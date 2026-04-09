# -*- coding: utf-8 -*-
"""
Главный скрипт: распознавание рукописных букв сетью Хопфилда.

Запуск:
    python main.py
"""

from sklearn.model_selection import train_test_split

from data_generation import generate_dataset, ALPHABET_UPPER
from experiments import run_experiments
from visualization import plot_results


def main():
    # ------------------------------------------------------------------
    # 1. Генерация датасета
    # ------------------------------------------------------------------
    print("Генерация датасета...")
    X, y, label_map = generate_dataset(
        letters=ALPHABET_UPPER,
        samples_per_letter=70,
        img_size=28,        # N = 28×28 = 784 нейрона
        font_size=20,
        noise_level=0.05,
        shift_range=2,
        rotation_range=10,
    )

    # ------------------------------------------------------------------
    # 2. Разбивка: 60% обучение / 20% валидация / 20% тест
    # ------------------------------------------------------------------
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp
    )

    print(f"\nРазмер выборок:")
    print(f"  Обучающая:     {X_train.shape[0]} примеров")
    print(f"  Валидационная: {X_val.shape[0]} примеров")
    print(f"  Тестовая:      {X_test.shape[0]} примеров")
    print(f"  Размер входа:  {X_train.shape[1]} (= {28}×{28} пикселей)")

    K = len(label_map)
    print(f"  Число классов: {K}")

    # ------------------------------------------------------------------
    # 3. Эксперименты с сетью Хопфилда
    # ------------------------------------------------------------------
    results = run_experiments(X_train, y_train, X_val, y_val, X_test, y_test, K)

    # ------------------------------------------------------------------
    # 4. Визуализация
    # ------------------------------------------------------------------
    plot_results(results, label_map, X_test, y_test)
    print("\nГотово. Графики сохранены в текущей директории.")


if __name__ == "__main__":
    main()
