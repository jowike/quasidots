from aeon.datasets import load_classification
from quasidots.classifier import QuasidoTSClassifier
from sktime.classification.kernel_based import RocketClassifier
from sklearn.metrics import accuracy_score


def run(dataset, operation="AT", num_kernels=1000, seed=42):
    X_train, y_train = load_classification(dataset, split="train", load_equal_length=True, load_no_missing=True)
    X_test, y_test = load_classification(dataset, split="test", load_equal_length=True, load_no_missing=True)

    quasidots = QuasidoTSClassifier(
        num_kernels=num_kernels,
        operation=operation,
        random_state=seed
    )

    rocket = RocketClassifier(
        num_kernels=num_kernels,
        random_state=seed
    )

    quasidots.fit(X_train, y_train)
    rocket.fit(X_train, y_train)

    return {
        "quasidots": accuracy_score(y_test, quasidots.predict(X_test)),
        "rocket": accuracy_score(y_test, rocket.predict(X_test))
    }


if __name__ == "__main__":
    result = run("ECG200")
    print(result)