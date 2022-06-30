import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # -1 for CPU 0, 1, etc for GPU(s)


import os
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard

from dataset import IteratorWithAug
import config
import Unet

from keras import backend as K


def jaccard_distance(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


# Set up datasets
x_train = [''] * 10000  # TODO: not used for this demo of random images... Later, set to file paths
y_train = [''] * 10000  # TODO: not used for this demo of random images... Later, set to file paths
x_val = [''] * 100  # TODO: not used for this demo of random images... Later, set to file paths
y_val = [''] * 100  # TODO: not used for this demo of random images... Later, set to file paths

train_generator = IteratorWithAug(
    image_paths=x_train,
    mask_paths=y_train,
    config=config,
    augmenter=config.AUGMENTER_TRAIN,
    mode='train',
    shuffle=True,
    batch_size=config.BATCH_SIZE,
)

val_generator = IteratorWithAug(
    image_paths=x_val,
    mask_paths=y_val,
    config=config,
    augmenter=config.AUGMENTER_VAL,
    mode='val',
    shuffle=False,
    batch_size=config.BATCH_SIZE,
)

# Set up log dir
datetime_str = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
LOG_DIR = os.path.join(r'.\logs', datetime_str)


# Build Model
net = Unet.Unet(outchns=config.CLASSES)
model = net.model

model.compile(
    optimizer=Adam(learning_rate=0.1),
    loss='mean_squared_error',
    metrics=[
        jaccard_distance
    ]
)


# Train
model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=config.EPOCHS,
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=[
        ModelCheckpoint(os.path.join(LOG_DIR, 'weights_{epoch:02d}.h5'), save_weights_only=False),
        TensorBoard(LOG_DIR, write_graph=False)
    ]
)
