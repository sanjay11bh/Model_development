import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DataAugmentor:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def add_spectra(self):
        X, y = self.X, self.y
        X_values = X.values
        y = y.values.flatten() if isinstance(y, (pd.Series, pd.DataFrame)) else y

        n = len(X_values)
        X_new, y_new = [], []

        for _ in range(n):
            i, j = np.random.choice(n, size=2, replace=False)
            X_new.append((X_values[i] + X_values[j]) / 2)
            y_new.append((y[i] + y[j]) / 2)

        X_aug = pd.DataFrame(np.vstack([X_values, X_new]), columns=X.columns)
        y_aug = pd.Series(np.concatenate([y, y_new]), name='target')
        return X_aug, y_aug

    def mixup(self, num_copies=2, alpha=0.4):
        X, y = self.X, self.y
        X_values = X.values
        y = y.values.flatten() if isinstance(y, (pd.Series, pd.DataFrame)) else y
        n = len(X_values)
        X_mix, y_mix = [], []

        for _ in range(num_copies):
            i, j = np.random.choice(n, size=2, replace=False)
            lam = np.random.beta(alpha, alpha)
            X_mix.append(lam * X_values[i] + (1 - lam) * X_values[j])
            y_mix.append(lam * y[i] + (1 - lam) * y[j])

        X_aug = pd.DataFrame(np.vstack([X_values] + [X_mix]), columns=X.columns)
        y_aug = pd.Series(np.concatenate([y, y_mix]), name='target')
        return X_aug, y_aug

    def spectral_shift(self, num_copies=2, max_shift=3):
        X, y = self.X, self.y
        X_values = X.values
        y = y.values.flatten() if isinstance(y, (pd.Series, pd.DataFrame)) else y
        n = len(X_values)
        X_shifted, y_shifted = [], []

        for _ in range(num_copies):
            for i in range(n):
                shift = np.random.randint(-max_shift, max_shift + 1)
                X_shifted.append(np.roll(X_values[i], shift))
                y_shifted.append(y[i])

        X_aug = pd.DataFrame(np.vstack([X_values] + [X_shifted]), columns=X.columns)
        y_aug = pd.Series(np.concatenate([y, y_shifted]), name='target')
        return X_aug, y_aug

    def gaussian_noise(self, num_copies=2, mean=0.0, std=0.01, random_state=None):
        X, y = self.X, self.y
        X_values = X.values
        y = y.values.flatten() if isinstance(y, (pd.Series, pd.DataFrame)) else y
        rng = np.random.RandomState(random_state)

        X_noisy = [X_values + rng.normal(loc=mean, scale=std, size=X_values.shape) for _ in range(num_copies)]
        y_noisy = [y for _ in range(num_copies)]

        X_aug = pd.DataFrame(np.vstack([X_values] + X_noisy), columns=X.columns)
        y_aug = pd.Series(np.concatenate([y] + y_noisy), name='target')
        return X_aug, y_aug