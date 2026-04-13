import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


print("load Fashion-MNIST Datasets...")
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(f"training set size: {x_train.shape[0]} ")
print(f"test set size: {x_test.shape[0]} ")

x_train_norm = x_train / 255.0
x_test_norm = x_test / 255.0

x_train_cnn = x_train_norm[..., np.newaxis]
x_test_cnn = x_test_norm[..., np.newaxis]

x_train_rf = x_train_norm.reshape(x_train_norm.shape[0], -1)
x_test_rf = x_test_norm.reshape(x_test_norm.shape[0], -1)


#CNN
print("\nBulid CNN model...")
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("start training CNN...")
start_cnn = time.time()
history = model.fit(x_train_cnn, y_train,
                    epochs=10,
                    validation_split=0.1,
                    batch_size=32,
                    verbose=1)
end_cnn = time.time()
cnn_train_time = end_cnn - start_cnn

# CNN evaluation
test_loss, test_acc = model.evaluate(x_test_cnn, y_test, verbose=0)
print(f"\nCNN test accuracy: {test_acc:.4f}, test_loss: {test_loss:.4f}")
print(f"CNN training time: {cnn_train_time:.2f} s")

y_pred_cnn = np.argmax(model.predict(x_test_cnn), axis=1)
print("\nCNN classification report:")
print(classification_report(y_test, y_pred_cnn, target_names=class_names))

cm_cnn = confusion_matrix(y_test, y_pred_cnn)
print("\nCNN confusion matrix (first 6 rows only):")
print(cm_cnn[:6])  

# Bulid random forest model...
print("\nBulid random forest model...")
rf = RandomForestClassifier(n_estimators=100,
                            random_state=42,
                            n_jobs=-1)  

print("Training random forest model...")
start_rf = time.time()
rf.fit(x_train_rf, y_train)
end_rf = time.time()
rf_train_time = end_rf - start_rf

y_pred_rf = rf.predict(x_test_rf)
rf_test_acc = accuracy_score(y_test, y_pred_rf)
print(f"\nRandom forest test accuracy: {rf_test_acc:.4f}")
print(f"Random forest training time: {rf_train_time:.2f} s")

print("\nRandom forest classification report:")
print(classification_report(y_test, y_pred_rf, target_names=class_names))

cm_rf = confusion_matrix(y_test, y_pred_rf)
print("\nRandom forest confusion matrix (first 6 rows only):")
print(cm_rf[:6])


# confusion matrix

def plot_confusion_matrix(cm, title, save_path=None):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

plot_confusion_matrix(cm_cnn, 'CNN confusion matrix', save_path='cnn_confusion_matrix.png')
plot_confusion_matrix(cm_rf, 'Random forest confusion matrix', save_path='rf_confusion_matrix.png')


# output key table

print("\n===== Model comparison =====")
print(f"{'Metrix':<20} {'CNN':<15} {'Random forest':<15}")
print("-" * 50)
print(f"{'Test Accuracy':<20} {test_acc:<15.4f} {rf_test_acc:<15.4f}")
print(f"{'Training time (s)':<20} {cnn_train_time:<15.2f} {rf_train_time:<15.2f}")
