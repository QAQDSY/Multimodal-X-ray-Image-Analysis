# lasso_selector.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from config import C_VALUES, THRESHOLD, OUTPUT_DIR, TOP_K
import os

def scale_features(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler

def lasso_selection(X_scaled, y, feature_names, prefix):
    pipe = Pipeline([
        ('selector', SelectFromModel(
            LogisticRegression(penalty='l1', solver='liblinear', random_state=42, max_iter=1000),
            threshold=THRESHOLD
        )),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    grid = GridSearchCV(pipe, {'selector__estimator__C': C_VALUES}, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_scaled, y)
    selector = grid.best_estimator_['selector']
    X_reduced = selector.transform(X_scaled)
    print(f"{prefix} after dimensionality reduction: {X_reduced.shape}")
    return selector, X_reduced, grid.best_params_

def plot_lasso_path(X, y, feature_names, title, save_path):
    Cs = np.logspace(-3, 2, 20)
    coefs = []
    for c in Cs:
        lr = LogisticRegression(penalty='l1', solver='liblinear', C=c, max_iter=1000)
        lr.fit(X, y)
        coefs.append(lr.coef_.flatten())
    coefs = np.array(coefs)

    plt.figure(figsize=(10, 6))
    for i in range(coefs.shape[1]):
        plt.plot(np.log10(Cs), coefs[:, i], lw=1)
    plt.xlabel("log10(C)"); plt.ylabel("Coefficient")
    plt.title(title); plt.grid(True)
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()

def get_top_k(selector, names, k=TOP_K):
    mask = selector.get_support()
    sel_names = np.array(names)[mask]
    sel_coefs = selector.estimator_.coef_.flatten()[mask]
    idx = np.argsort(np.abs(sel_coefs))[::-1][:k]
    return [(sel_names[i], sel_coefs[i]) for i in idx]

def plot_combined_bar(top1, top2, save_path):
    items = [(f"Rad_{n}", c) for n, c in top1] + [(f"DL_{n}", c) for n, c in top2]
    names, values = zip(*items)
    plt.figure(figsize=(12, 6))
    bars = plt.bar(names, values)
    for bar, v in zip(bars, values):
        bar.set_color('blue' if v >= 0 else 'red')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("LASSO Coefficient"); plt.title("Top 10 Radiomics + DL Features")
    plt.grid(True, axis='y', alpha=0.7)
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()