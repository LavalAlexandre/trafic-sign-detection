from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import warnings
import joblib
from src.image import img
from sklearn.metrics import confusion_matrix
import seaborn as sns
import cv2


def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, normalize="true", labels=labels)
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


class model:
    def __init__(self, seed=42, n_jobs=-1, standard_size=(64, 64)):
        self.seed = seed
        # Must match the standard_size of the dataset the classifier was trained on
        self.standard_size = standard_size
        self.classifier = None
        self.name = None
        self.n_jobs = n_jobs

    def save(self, path, name="model"):
        model_path = path + f"/{name}" + ".pkl"
        joblib.dump(self.classifier, model_path)

    def load(self, model_path):
        self.classifier = joblib.load(model_path)
        self.name = type(self.classifier).__name__
        return self

    def predict_window(self, window):
        window = img(standard_size=self.standard_size, window=window)
        return self.classifier.predict([window.data])

    def predict_proba_window(self, window):
        window = img(standard_size=self.standard_size, window=window)
        return self.classifier.predict_proba([window.data])

    def predict(self, window):
        """Common classifier interface used by src.detection: returns (label, confidence)."""
        probabilities = self.predict_proba_window(window)[0]
        best = np.argmax(probabilities)
        return self.classifier.classes_[best], probabilities[best]

    def train_svm(self, train_data, max_iter=1000, verbose=0):
        self.name = "SVM"
        X = [img.data for img in train_data.images]
        y = [img.label for img in train_data.images]

        print(f"Number of nan values in X: {np.isnan(X).sum()}")

        self.classifier = SVC(
            kernel="linear",
            shrinking=True,
            random_state=self.seed,
            max_iter=max_iter,
            verbose=verbose,
            class_weight="balanced",
            probability=True,
            C=10
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.seed
        )
        self.classifier.fit(X_train, y_train)

        y_test_pred = self.classifier.predict(X_test)
        print(f"Accuracy on held-out train split: {accuracy_score(y_test, y_test_pred)}")
        print(classification_report(y_test, y_test_pred))
        plot_confusion_matrix(y_test, y_test_pred, self.classifier.classes_)

    def train_elastic_net(self, train_data, max_iter=1000, verbose=0):
        self.name = "Elastic Net"
        X = [img.data for img in train_data.images]
        y = [img.label for img in train_data.images]

        self.classifier = LogisticRegression(
            random_state=self.seed,
            max_iter=max_iter,
            verbose=verbose,
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.seed
        )
        self.classifier.fit(X_train, y_train)
        # on test data accuracy_score
        print(
            f"Accuracy on test data: {accuracy_score(y_test, self.classifier.predict(X_test))}"
        )

    def grid_search_svm(self, train_data, max_iter=1000, verbose=0):
        self.name = "grid linear svm.SVC"
        warnings.filterwarnings("ignore")

        # Extract data and labels from train_data
        X = np.array([img.data for img in train_data.images])
        y = np.array([img.label for img in train_data.images])

        # Define the parameter grid for grid search
        tuned_parameters = {
            "kernel": ["linear"],
            "C": [1, 10, 100, 1000],
            "class_weight": [None, "balanced"],
            "max_iter": [max_iter],
            "shrinking": [True, False],
        }

        grid_search = GridSearchCV(
            estimator=SVC(probability=True), param_grid=tuned_parameters, cv=5, verbose=verbose, n_jobs=-1
        )
        grid_search.fit(X, y)

        self.classifier = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_

        print(f"Best parameters found: {self.best_params}")
        print(f"Best cross-validation score: {self.best_score:.4f}")
        warnings.filterwarnings("default")

    def train_lr(self, train_data, max_iter=1000, verbose=0):
        self.name = "Logistic Regression"
        X = [img.data for img in train_data.images]
        y = [img.label for img in train_data.images]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.seed
        )
        self.classifier = LogisticRegression(
            random_state=self.seed,
            max_iter=max_iter,
            verbose=verbose,
            n_jobs=self.n_jobs,
        )
        print(f"Start training {self.name} with max_iter {max_iter}")
        self.classifier.fit(X_train, y_train)
        # on test data accuracy_score
        print(
            f"Accuracy on test data: {accuracy_score(y_test, self.classifier.predict(X_test))}"
        )

    def train_dual_lr(self, train_data, max_iter=1000, verbose=0):
        self.name = "dual liblinear Logistic Regression"
        X = [img.data for img in train_data.images]
        y = [img.label for img in train_data.images]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.seed
        )
        self.classifier = LogisticRegression(
            random_state=self.seed,
            max_iter=max_iter,
            verbose=verbose,
            dual=True,
            solver="liblinear",
        )
        self.classifier.fit(X_train, y_train)
        # on test data accuracy_score
        print(
            f"Accuracy on test data: {accuracy_score(y_test, self.classifier.predict(X_test))}"
        )

    def train_xgboost(self, train_data, verbose=0):
        # Optional dependency: `uv sync --extra xgboost`
        from xgboost import XGBClassifier

        self.name = "XGBoost"
        X = [img.data for img in train_data.images]
        y = [img.label for img in train_data.images]
        y = train_data.LabelEncoder.transform(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.seed
        )
        self.classifier = XGBClassifier(
            n_estimators=100, random_state=self.seed, verbosity=verbose
        )
        self.classifier.fit(X_train, y_train)
        # on test data accuracy_score
        print(
            f"Accuracy on test data: {accuracy_score(y_test, self.classifier.predict(X_test))}"
        )

    def train_OneVsRest(self, train_data, max_iter=100, verbose=0):
        self.name = "OneVsRest"
        X = [img.data for img in train_data.images]
        y = [img.label for img in train_data.images]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.seed
        )
        base_clf = LogisticRegression(
            random_state=self.seed, max_iter=max_iter, verbose=verbose
        )
        self.classifier = OneVsRestClassifier(base_clf)
        self.classifier.fit(X_train, y_train)
        # on test data accuracy_score
        print(
            f"Accuracy on test data: {accuracy_score(y_test, self.classifier.predict(X_test))}"
        )

    def evaluate(self, val_data, show_errors=False):
        X = [img.data for img in val_data.images]
        y = [img.label for img in val_data.images]
        if self.name == "XGBoost":
            y = val_data.LabelEncoder.transform(y)
        y_pred = self.classifier.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print(f"Accuracy on validation data: {accuracy}")
        print(classification_report(y, y_pred))
        plot_confusion_matrix(y, y_pred, self.classifier.classes_)

        if show_errors:
            for image, label, pred in zip(val_data.images, y, y_pred):
                if label != pred:
                    plt.imshow(cv2.cvtColor(cv2.imread(image.path + image.name), cv2.COLOR_BGR2RGB))
                    plt.title(f"Correct label: {label}, Predicted label: {pred}")
                    plt.show()

        return accuracy

    def plot_learning_curve(self, train_data, verbose=0):
        X = [img.data for img in train_data.images]
        y = [img.label for img in train_data.images]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.seed
        )

        # Generate learning curve data
        train_sizes, train_scores, test_scores = learning_curve(
            self.classifier, X_train, y_train, verbose=verbose, cv=None
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure()
        plt.title("Learning Curve")
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.grid()
        plt.fill_between(
            train_sizes,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.1,
            color="r",
        )
        plt.fill_between(
            train_sizes,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.1,
            color="g",
        )
        plt.plot(
            train_sizes, train_scores_mean, "o-", color="r", label="Training score"
        )
        plt.plot(
            train_sizes,
            test_scores_mean,
            "o-",
            color="g",
            label="Cross-validation score",
        )
        plt.legend(loc="best")
        plt.show()
