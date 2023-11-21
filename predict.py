import tensorflow as tf
keras=tf.keras
import numpy as np
import matplotlib.pyplot as plt

def predictans(image):
    model = tf.keras.models.load_model('MACHINELEARNING/DRAW_PRED/canvasmodel')
    num=model.predict(image)
    num=np.argmax(num)

    # image2=image.reshape(28,28)
    # plt.imshow(image2, cmap='gray')
    # plt.show()

    return num