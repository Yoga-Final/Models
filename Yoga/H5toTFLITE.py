
import tensorflow as tf
#from tensorflow.contrib import lite

new_model= tf.keras.models.load_model(filepath="Modals/Final(2).h5")
tflite_converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
tflite_model = tflite_converter.convert()
open("Modals/Final(2).tflite", "wb").write(tflite_model)
print("done")
