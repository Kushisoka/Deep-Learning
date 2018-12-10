import os
import sys
import pprint
import tensorflow as tf
from tensorflow.python import debug as tf_debug

from Net import ResBlock
from Net import SeparableResBlock
from Net import EluResBlock
from Net import ResNetIntro
from Net import ResNetStage
from Net import FirstResNetStage

from Net import PreActivatedResNetStage
from Net import FirstPreActivatedResNetStage

from tensorflow.keras import Input
from tensorflow.keras import layers
from tensorflow.keras.models import Model as md
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers

from keras.preprocessing.image import ImageDataGenerator
import time
import numpy as np

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint




if 'COLAB_TPU_ADDR' not in os.environ:
  print('ERROR: Not connected to a TPU runtime; please see the first cell in this notebook for instructions!')
else:
  tpu_address = 'grpc://' + os.environ['COLAB_TPU_ADDR']
  print ('TPU address is', tpu_address)
  
  with tf.Session(tpu_address) as session:
    devices = session.list_devices()
    #tf.keras.backend.set_session(tf_debug.LocalCLIDebugWrapperSession(session))
    #session = tf_debug.TensorBoardDebugWrapperSession(session)

  print('TPU devices:')
  pprint.pprint(devices)
  
#tf.keras.backend.get_session().run(tf.global_variables_initializer())
#tf.InteractiveSession(graph=tf.Graph())

"""# Build"""


data_format = None
regulizer = None
padding ='same'
use_bias = True
avg = True
channels = 1


#regulizer = regularizers.l2(0.02)

name = "Autentic ResNet"

inpt = Input(shape=(224, 224, 3), name='Input', )

comun = ResNetIntro(inpt, kernel_regularizer = regulizer)

comun = FirstResNetStage(comun,size = 64, N = 3, kernel_regularizer = regulizer)

comun = ResNetStage(comun,size = 128, N = 4, kernel_regularizer = regulizer)
comun = ResNetStage(comun,size = 256, N = 6, kernel_regularizer = regulizer)
comun = ResNetStage(comun,size = 512, N = 3, kernel_regularizer = regulizer)
comun = layers.GlobalAveragePooling2D()(comun)

output = layers.Dense(59, activation='softmax')(comun)

keras_model = md(inpt, output)

"""# Keras to TPU"""

TPU_WORKER = 'grpc://' + os.environ['COLAB_TPU_ADDR']

tpu_model = tf.contrib.tpu.keras_to_tpu_model(
    keras_model,
    strategy=tf.contrib.tpu.TPUDistributionStrategy(
        tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)))

"""# Compile"""

tpu_model.compile(optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1), loss='categorical_crossentropy', metrics=['acc'])

"""# Callbacks"""

log_dir = 'logs/' + name + '/'
tensorboard = TensorBoard(log_dir=log_dir)

lr_reductor = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)

filepath = "weights-epoch{epoch:02d}-acc{val_acc:.2f}.hdf5"
model_checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_weights_only=False, save_best_only=True, mode='max', period=1)


"""# Data Augmentation (Raw Data)"""

val_batch_size = 64*8

start_time = time.time()

dataset = np.load("../gdrive/My Drive/TPU/dataset.npy")

dataset_labels = np.load("../gdrive/My Drive/TPU/dataset.npy")

validation_split = 0.2

index_val = int(validation_split*len(dataset))

val_x = dataset[:index_val]
val_y = dataset_labels[:index_val]

val_size = len(val_x)

dataset = dataset[index_val:]
dataset_labels = dataset_labels[index_val:]

train_size = len(dataset)

train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(dataset, dataset_labels, batch_size=val_batch_size)
validation_generator = test_datagen.flow(val_x, val_y, batch_size=val_batch_size)

print("Time: " + str(time.strftime("%Hh%Mm%Ss", time.gmtime((time.time() - start_time)))))

"""# Train"""

batch_size=128*8
epochs_number = 200

print("INICIO DEL ENTRENAMIENTO")

start_time = time.time()

history = tpu_model.fit_generator(train_generator,steps_per_epoch=8,epochs=epochs_number, use_multiprocessing=False ,validation_data=validation_generator,validation_steps=8, callbacks=[tensorboard, model_checkpoint])

print("FIN DEL ENTRENAMIENTO")

print("Time:", str(time.strftime("%Hh%Mn%Ss", time.gmtime((time.time() - start_time)))))

