import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import config
from core.classifier import save_classifier


def generate_training_data(n_samples=5000):
    np.random.seed(42)
    samples_per_class = n_samples // 3

    def class_samples(ndvi_m, red_m, green_m, tex_m, moist_m, label, n):
        rows = []
        for _ in range(n):
            rows.append([
                np.random.normal(*ndvi_m),
                np.random.normal(*red_m),
                np.random.normal(*green_m),
                np.random.normal(*tex_m),
                np.random.normal(*moist_m),
            ])
        return rows, [label] * n

    xh, yh = class_samples((0.75, 0.05), (0.2, 0.05), (0.8, 0.05), (0.1, 0.02), (0.7, 0.1), config.CELL_HEALTHY, samples_per_class)
    xe, ye = class_samples((0.45, 0.08), (0.5, 0.1), (0.5, 0.1), (0.4, 0.1), (0.4, 0.1), config.CELL_EARLY, samples_per_class)
    xs, ys = class_samples((0.20, 0.06), (0.8, 0.05), (0.2, 0.05), (0.8, 0.1), (0.1, 0.05), config.CELL_SEVERE, samples_per_class)

    X = np.clip(np.vstack((xh, xe, xs)), 0.0, 1.0)
    y = np.hstack((yh, ye, ys))
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]


def train_model(X, y):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    print("Training Accuracy:", accuracy_score(y, y_pred))
    print(classification_report(y, y_pred, target_names=["Healthy", "Early", "Severe"]))
    return clf


def train_and_save():
    X, y = generate_training_data(5000)
    clf = train_model(X, y)
    save_classifier(clf)
    return clf


if __name__ == "__main__":
    train_and_save()
