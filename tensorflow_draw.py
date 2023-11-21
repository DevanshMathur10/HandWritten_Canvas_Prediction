import tensorflow as tf
keras=tf.keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,BatchNormalization
from keras import Sequential

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

model=Sequential()
model.add(Conv2D(32,(5,5),activation='relu',input_shape=(28,28,1)))
model.add(MaxPooling2D((3,3)))
model.add(BatchNormalization())
model.add(Conv2D(64,(5,5),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(10))

model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
model.summary()
history=model.fit(x_train,y_train,epochs=10,batch_size=30)

test_loss, test_acc = model.evaluate(x_test,  y_test)
print(f"Theaccuracy on test set is : {(test_acc*100):.6f}")

# model = tf.keras.models.load_model('MACHINELEARNING/DRAW_PRED/canvasmodel')

x=int(input("Choose an image between 1-9999 : "))
while x!=0:
    image=x_train[x]
    plt.imshow(image, cmap='gray')
    num=model.predict(image.reshape(1,28,28,1))
    num=np.argmax(num)
    plt.title(f"My neural network predicts {num} !")
    plt.show()
    x=int(input("Choose an image between 1-9999 : ")) 