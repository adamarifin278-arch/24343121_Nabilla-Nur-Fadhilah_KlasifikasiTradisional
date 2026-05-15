import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_digits
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, learning_curve, GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score
)

# =========================
# 1. LOAD DATASET
# =========================
digits = load_digits()

X = digits.data[:1000]
y = digits.target[:1000]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print("Dataset loaded:", X.shape)


# =========================
# 2. KNN EXPERIMENT
# =========================
def knn_experiment():
    print("\n===== KNN EXPERIMENT =====")

    k_values = [1, 3, 5, 7, 9, 11]
    metrics = ['euclidean', 'manhattan']

    results = []

    for metric in metrics:
        for k in k_values:

            model = KNeighborsClassifier(n_neighbors=k, metric=metric)

            cv = cross_val_score(model, X_train, y_train, cv=5)

            start = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start

            start = time.time()
            y_pred = model.predict(X_test)
            pred_time = time.time() - start

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')

            results.append([k, metric, acc, f1, cv.mean(), train_time, pred_time])

            print(f"K={k}, metric={metric}, acc={acc:.4f}, cv={cv.mean():.4f}")

    return results


# =========================
# 3. SVM EXPERIMENT
# =========================
def svm_experiment():
    print("\n===== SVM EXPERIMENT =====")

    kernels = ['linear', 'poly', 'rbf']
    C_values = [0.1, 1, 10, 100]

    results = []

    for kernel in kernels:
        for C in C_values:

            model = svm.SVC(kernel=kernel, C=C)

            cv = cross_val_score(model, X_train, y_train, cv=5)

            start = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start

            start = time.time()
            y_pred = model.predict(X_test)
            pred_time = time.time() - start

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')

            results.append([kernel, C, acc, f1, cv.mean(), train_time, pred_time])

            print(f"{kernel}, C={C}, acc={acc:.4f}, cv={cv.mean():.4f}")

    return results


# =========================
# 4. CONFUSION MATRIX
# =========================
def plot_cm(model, title):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


# =========================
# 5. PCA + DECISION BOUNDARY (SVM)
# =========================
def plot_decision_boundary():

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train)

    model = svm.SVC(kernel='rbf', C=10)
    model.fit(X_pca, y_train)

    x_min, x_max = X_pca[:,0].min()-1, X_pca[:,0].max()+1
    y_min, y_max = X_pca[:,1].min()-1, X_pca[:,1].max()+1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.2),
        np.arange(y_min, y_max, 0.2)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8,6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

    scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=y_train, cmap='tab10', edgecolor='k')
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title("SVM Decision Boundary (PCA 2D)")
    plt.show()


# =========================
# 6. LEARNING CURVE
# =========================
def plot_learning_curve():

    model = svm.SVC(kernel='rbf', C=10)

    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train,
        cv=5,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5)
    )

    plt.figure(figsize=(8,6))
    plt.plot(train_sizes, train_scores.mean(axis=1), label="Train Accuracy")
    plt.plot(train_sizes, test_scores.mean(axis=1), label="CV Accuracy")

    plt.title("Learning Curve SVM")
    plt.xlabel("Training Size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.show()


# =========================
# 7. GRID SEARCH (SVM OPTIMIZATION)
# =========================
def grid_search_svm():

    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['rbf', 'linear'],
        'gamma': [0.001, 0.01, 0.1]
    }

    grid = GridSearchCV(
        svm.SVC(),
        param_grid,
        cv=3,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("\nBEST PARAMS:", grid.best_params_)
    print("BEST SCORE:", grid.best_score_)

    return grid.best_estimator_


# =========================
# 8. RUN ALL
# =========================

knn_results = knn_experiment()
svm_results = svm_experiment()

best_svm = grid_search_svm()

plot_cm(best_svm, "Confusion Matrix - Best SVM")
plot_decision_boundary()
plot_learning_curve()

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, best_svm.predict(X_test)))