from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import time


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)


shallow_model = Sequential([
    Dense(100, activation='relu', input_shape=(784,)),
    Dense(50, activation='relu'),
    Dense(100, activation='relu'),
    Dense(10, activation='softmax')
])


shallow_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

start_time = time.time()
history = shallow_model.fit(x_train, y_train_cat, validation_split=0.2,
                            epochs=50, batch_size=128, verbose=2)
end_time = time.time()


test_loss, test_acc = shallow_model.evaluate(x_test, y_test_cat, verbose=0)
print(f"\nShallow Model Test Accuracy: {test_acc:.4f}")
print(f"\nTraining Time: {end_time - start_time:.2f} δευτερόλεπτα")


plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Shallow Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Shallow Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
