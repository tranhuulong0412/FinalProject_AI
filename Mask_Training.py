#-----------------------------------------------------IMPORT LIBRARIES--------------------------------------------------------------------------------------------
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot as plt 
from sklearn import preprocessing
from tensorflow.keras.models import Sequential

#--------------------------------------------------------DATASETS--------------------------------------------------------------------------------------------
train_datagen=ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

training_set=train_datagen.flow_from_directory(r'C:\Users\admin\Desktop\ASUS\AI_FINAL\Train',target_size=(150,150), batch_size=32, class_mode='categorical')
testing_set=test_datagen.flow_from_directory(r'C:\Users\admin\Desktop\ASUS\AI_FINAL\Test',target_size=(150,150), batch_size=32, class_mode='categorical')

training_set.class_indices
testing_set.class_indices

#-------------------------------------------------------CREATE MODEL--------------------------------------------------------------------------------------------
model=Sequential()
model.add(Conv2D(32,(3,3), activation='relu', kernel_initializer='he_uniform', padding='same',input_shape=(150,150,3))) 
model.add(LeakyReLU(alpha = 0.1))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64,(3,3), activation='relu', kernel_initializer='he_uniform', padding='same')) 
model.add(LeakyReLU(alpha = 0.1))

model.add(MaxPooling2D(2,2))
model.add(Conv2D(64,(3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))

model.add(LeakyReLU(alpha = 0.1))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(64,activation='relu',kernel_initializer='he_uniform'))
model.add(Dense(2,activation='softmax'))
model.summary()

#-------------------------------------------------------COMPILE MODEL--------------------------------------------------------------------------------------------
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history=model.fit(training_set,epochs=50,batch_size=128,verbose=1,validation_data=testing_set)

#-----------------------------------------------------EVALUATE ACCURACY--------------------------------------------------------------------------------------------
Score=model.evaluate(training_set,verbose=0)
print('Loss', Score[0])
print('Accuracy', Score[1])

#--------------------------------------------------------SAVE MODEL------------------------------------------------------------------------------------------------------------------------------------
model.save("MaskDetection.h5")
