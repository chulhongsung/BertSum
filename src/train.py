import tensorflow as tf
from tensorflow import keras as K

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

class BertSumLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate):
        self.initial_learning_rate = initial_learning_rate

    def __call__(self, step):
        return 2 * tf.math.exp(-3.0) * tf.math.minimum((step+1)**(-0.5),(step+1) * (100.0)**(-1.5))
                                                                
optimizer = tf.keras.optimizers.Adam(learning_rate = BertSumLRSchedule(9.95e-05))

# BATCH_SIZE = 5

# batch_data = sample_data.batch(BATCH_SIZE)

# EPOCHS = 100

# for epoch in range(EPOCHS):
#     for iis, ttis, ams, cis, cirt, ground_truth in iter(batch_data):
#         train_step(ebs, iis, ttis, ams, cis, ,cirt, ground_truth)
#     template = 'EPOCH: {}, Train Loss: {}'
#     print(template.format(epoch+1, train_loss.result()))
