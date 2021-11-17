import tensorflow as tf
from tensorflow import keras as K

optimizer = tf.keras.optimizers.Adam(learning_rate = 0.00001)

train_loss = K.metrics.Mean(name='train_loss')

class RaggedBinaryCrossEntropy(K.losses.Loss):
    @tf.function
    def call(self, y_true, y_pred):
        losses = []
        for i in range(y_true.shape[0]):
            losses.append(K.losses.binary_crossentropy(y_true[i], y_pred[i]))
        return tf.reduce_mean(losses)    
    
rbc = RaggedBinaryCrossEntropy()

@tf.function
def train_step(model, iis, ttis, ams, cis, cirt, ground_truth):
    with tf.GradientTape(persistent=True) as tape:
        predicted = model((iis, ttis, ams), cis, cirt)
        loss = rbc(ground_truth, predicted)
    grad = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grad, model.trainable_weights))
    train_loss(loss) 
    
    
# EPOCHS = 100

# for epoch in range(EPOCHS):
#     for iis, ttis, ams, cis, cirt, ground_truth in iter(batch_data):
#         train_step(ebs, iis, ttis, ams, cis, ,cirt, ground_truth)
#     template = 'EPOCH: {}, Train Loss: {}'
#     print(template.format(epoch+1, train_loss.result()))
