"""
Сеть Хопфилда для распознавания рукописных букв.

Архитектура:
  - N нейронов (= размер входного вектора, например 784 для 28×28)
  - Матрица весов W (N×N), симметричная, нулевая диагональ
  - Обучение: правило Хебба (однократный проход по паттернам)
  - Вспоминание: асинхронное обновление до сходимости

Параметры экспериментов:
  - threshold (θ) — порог передаточной функции нейрона
  - Ns           — число хранимых паттернов (ёмкость сети)
"""

import numpy as np


class HopfieldNetwork:
    """
    Классическая дискретная сеть Хопфилда.

    Parameters
    ----------
    n_neurons : int
        Число нейронов (= размерность входного вектора).
    threshold : float
        Порог θ передаточной функции sign(u - θ).
        При θ=0 — стандартная функция знака.
    max_iter : int
        Максимальное число итераций асинхронного обновления.
    """

    def __init__(self, n_neurons: int, threshold: float = 0.0, max_iter: int = 20):
        self.n_neurons = n_neurons
        self.threshold = threshold
        self.max_iter = max_iter
        self.W = np.zeros((n_neurons, n_neurons), dtype=np.float64)
        self.patterns: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Обучение (правило Хебба)
    # ------------------------------------------------------------------

    def train(self, patterns: np.ndarray) -> None:
        """
        Запоминает паттерны методом Хебба.

        Parameters
        ----------
        patterns : np.ndarray, shape (P, N)
            Биполярные паттерны (+1 / -1).
        """
        P, N = patterns.shape
        assert N == self.n_neurons, "Размер паттерна не совпадает с числом нейронов"

        self.W = np.zeros((N, N), dtype=np.float64)
        for xi in patterns:
            self.W += np.outer(xi, xi)

        # Нормировка и обнуление диагонали
        self.W /= N
        np.fill_diagonal(self.W, 0.0)

        self.patterns = list(patterns)

    # ------------------------------------------------------------------
    # Вспоминание (асинхронное обновление)
    # ------------------------------------------------------------------

    def recall(self, x: np.ndarray) -> np.ndarray:
        """
        Восстанавливает паттерн из зашумлённого входа.

        Parameters
        ----------
        x : np.ndarray, shape (N,)
            Зашумлённый биполярный вектор.

        Returns
        -------
        np.ndarray, shape (N,)
            Стабильное состояние сети.
        """
        state = x.copy().astype(np.float64)
        for _ in range(self.max_iter):
            prev = state.copy()
            # Случайный порядок обновления нейронов (асинхронный режим)
            order = np.random.permutation(self.n_neurons)
            for i in order:
                u_i = self.W[i] @ state
                state[i] = 1.0 if u_i >= self.threshold else -1.0
            if np.array_equal(state, prev):
                break
        return state

    def recall_batch(self, X: np.ndarray) -> np.ndarray:
        """Вспоминание для батча векторов."""
        return np.array([self.recall(x) for x in X])

    # ------------------------------------------------------------------
    # Классификация: ближайший запомненный паттерн
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray, stored_labels: np.ndarray) -> np.ndarray:
        """
        Классифицирует входные векторы.

        После recall сравниваем результат с каждым запомненным паттерном
        по расстоянию Хэмминга и возвращаем метку ближайшего.

        Parameters
        ----------
        X : np.ndarray, shape (M, N)
        stored_labels : np.ndarray, shape (P,)
            Метки классов для каждого запомненного паттерна.

        Returns
        -------
        np.ndarray, shape (M,)
        """
        recalled = self.recall_batch(X)
        patterns_arr = np.array(self.patterns)  # (P, N)

        predictions = []
        for r in recalled:
            # Расстояние Хэмминга = число несовпадающих битов
            dists = np.sum(patterns_arr != r, axis=1)
            predictions.append(stored_labels[np.argmin(dists)])

        return np.array(predictions)

    # ------------------------------------------------------------------
    # Энергия сети
    # ------------------------------------------------------------------

    def energy(self, x: np.ndarray) -> float:
        """E = -½ xᵀ W x"""
        return -0.5 * float(x @ self.W @ x)
