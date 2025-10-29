import tensorflow as tf

mse = tf.keras.losses.MeanSquaredError()
y_true = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
y_pred = [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]

loss = mse(y_true, y_pred)
print("MSE Loss:", loss.numpy())
