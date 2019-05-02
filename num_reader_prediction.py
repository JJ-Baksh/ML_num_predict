import tensorflow as tf
# import matplotlib.pyplot as plt
import numpy as np
from tkinter import *
from PIL import Image, ImageDraw
import PIL

new_model = tf.keras.models.load_model('epic_num_reader.model')  # load model

# interface to ask user to draw a number


def answer_function():
    # image.save("number.jpg")
    image_data = np.array(image)

    image_data = tf.keras.utils.normalize(image_data, axis=0).reshape(1, -1)

# print(image_data)
# print(len(image_data))
# print(image_data.shape)
# print(image_data.shape[0])

    prediction = new_model.predict([image_data])
    answer = Label(root, text=("The number is a " + str(np.argmax(prediction))))
    answer.grid(row=3, column=0)


def free_draw(event):
    canvas.create_oval(event.x, event.y, event.x+25, event.y+25, width=0, fill='black')
    draw.ellipse((event.x*ratio, event.y*ratio, (event.x+40)*ratio, (event.y+40)*ratio), width=28, fill='white')


def clear_function():
    canvas.delete("all")
    draw.rectangle((0, 0, 28, 28), fill="black")


root = Tk()

label = Label(root, text="Draw any number for the algorithm to predict...", compound=TOP)
label.grid(row=0, column=0)
clear = Button(root, text='Clear', command=clear_function)
clear.grid(row=0, column=1)
canvas = Canvas(root, width=400, height=400, bg="white")
canvas.grid(row=1, column=0, columnspan=2)

ratio = 28/400

image = PIL.Image.new('L', (28, 28), 'black')
draw = ImageDraw.Draw(image)
draw.rectangle((0, 0, 28, 28), fill="black")

canvas.bind("<B1-Motion>", free_draw)
submit = Button(root, text='Submit', command=answer_function)
submit.grid(row=3, column=1)

root.mainloop()

'''
mnist = tf.keras.datasets.mnist  # 28*28 images of hand-written digits 0-9
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = tf.keras.utils.normalize(x_test, axis=1).reshape(x_test.shape[0], -1)

# using the model to predict from test data
predictions = new_model.predict([x_test])
# print(predictions)
place = 1
print(np.argmax(predictions[place]))
# to show the image of the prediction
plt.imshow(x_test[place].reshape(28, 28), cmap='gray')
plt.show()
'''