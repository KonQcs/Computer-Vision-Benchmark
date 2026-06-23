import argparse
import time

import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


def build_model() -> Sequential:
    return Sequential(
        [
            Dense(100, activation="relu", input_shape=(784,)),
            Dense(50, activation="relu"),
            Dense(100, activation="relu"),
            Dense(10, activation="softmax"),
        ]
    )


def load_dataset():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)
    return x_train, y_train_cat, x_test, y_test_cat


def plot_history(history) -> None:
    plt.plot(history.history["accuracy"], label="Train Acc")
    plt.plot(history.history["val_accuracy"], label="Val Acc")
    plt.title("Shallow Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Shallow Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the shallow Fashion-MNIST model.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--no-plots", action="store_true", help="Skip matplotlib windows.")
    args = parser.parse_args()

    x_train, y_train_cat, x_test, y_test_cat = load_dataset()
    model = build_model()
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    start_time = time.time()
    history = model.fit(
        x_train,
        y_train_cat,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
    )
    elapsed = time.time() - start_time

    test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
    print(f"Shallow Model Test Loss: {test_loss:.4f}")
    print(f"Shallow Model Test Accuracy: {test_acc:.4f}")
    print(f"Training Time: {elapsed:.2f} seconds")

    if not args.no_plots:
        plot_history(history)


if __name__ == "__main__":
    main()
