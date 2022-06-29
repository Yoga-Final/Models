from keras.layers import Dense,Conv2D,Dropout,MaxPooling2D,Flatten
from keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from matplotlib import pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
train_masks_name=os.listdir("New Yoga Dataset/Train/Adho Mukha Shvanasana") 
train_nomasks_name=os.listdir("New Yoga Dataset/Train/z")
'''


train_dg=ImageDataGenerator(rescale=1./255,horizontal_flip=True,zoom_range=0.5,validation_split=0.1,rotation_range=5)
test_dg=ImageDataGenerator(rescale=1./255)
valid_dg=ImageDataGenerator(rescale=1./255)
train_g=train_dg.flow_from_directory("New Yoga Dataset/Main1",target_size=(150,150),subset="training") #Train
test_g=test_dg.flow_from_directory("New Yoga Dataset/Main1",target_size=(150,150))
valid_g=train_dg.flow_from_directory("New Yoga Dataset/Main1",target_size=(150,150),subset="validation")
print(len(train_g))
print(train_g.class_indices)
print(train_g.image_shape)







m=Sequential()
m.add(Conv2D(32,strides=(3,3), kernel_size=(3,3),activation=tf.nn.relu,padding="SAME",input_shape=train_g.image_shape))
m.add(MaxPooling2D(pool_size=(2,2)))
#m.add(Dropout())


m.add(Conv2D(32, kernel_size=(3,3),activation=tf.nn.relu,padding="SAME"))
m.add(MaxPooling2D(pool_size=(2,2)))


m.add(Flatten())
m.add(Dense(256,activation=tf.nn.relu))
m.add(Dropout(0.5))   ###### 7
m.add(Dense(256,activation=tf.nn.relu))
m.add(Dropout(0.5))   ###### 7
m.add(Dense(256,activation=tf.nn.relu))
#m.add(Dropout())
#m.add(Dense(256,activation=tf.nn.relu))

m.add(Dense(28,activation=tf.nn.softmax))   

m.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
f=m.fit(train_g,epochs=200,validation_data=valid_g,batch_size=10)
print(m.evaluate(test_g))

accuracy_train = f.history['accuracy']
accuracy_val = f.history['val_accuracy']
epochs = range(1,201)
fig=plt.figure("Figure")
plt.plot(epochs, accuracy_train, 'g', label='Training accuracy')
plt.plot(epochs,accuracy_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()
m.save("Modals/Final(2).h5")
print("done")