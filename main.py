import tkinter as tk
from PIL import Image, ImageDraw
from predict import predictans
import numpy as np
from PIL import ImageOps
import os

def printpred():
    im = Image.open('MACHINELEARNING/DRAW_PRED/image.jpg').convert("L")
    im1=im.resize((28,28))
    im2 = ImageOps.invert(im1)
    # im2.show()
    arr=np.array(im2.getdata()).reshape(1,28,28,1)
    num=predictans(arr)
    print(num)
    return num

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Drawing App")
        self.root.geometry("500x500")

        self.tlabel = tk.Label(root, text="HANDWRITTEN DIGIT PREDICTOR",font=('Consolas',16))
        self.tlabel.place(relx=0.5, rely=0.1, anchor="center")

        canvas_frame = tk.Frame(root)
        canvas_frame.place(relx=0.5, rely=0.5, anchor="center")

        self.canvas = tk.Canvas(canvas_frame, width=275, height=275, bg="white")
        self.canvas.pack(expand=tk.YES, fill=tk.BOTH)

        self.canvas.bind("<B1-Motion>", self.paint)

        button_frame = tk.Frame(root)
        button_frame.place(relx=0.5, rely=0.8, anchor="center")

        self.convert_button = tk.Button(button_frame, text="Predict", command=self.convert_and_save)
        self.convert_button.pack(side=tk.LEFT)

        self.clear_button = tk.Button(button_frame, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=tk.RIGHT)

        self.result_label = tk.Label(root, text="",font=('Century Gothic',12))
        self.result_label.place(relx=0.5, rely=0.9, anchor="center")

        self.image = Image.new("L", (300, 300), color="white")
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=8)
        self.draw.line([x1, y1, x2, y2], fill="black", width=8)

    def convert_and_save(self):

        self.image.save("MACHINELEARNING/DRAW_PRED/image.jpg")
        print("Image saved as image.jpg")

        result_text = "The predicted value with my Neural Network is : {}".format(self.printpred())
        self.result_label.config(text=result_text)
    
    def clear_canvas(self):
        # Clear the canvas and the image
        self.canvas.delete("all")
        self.image = Image.new("L", (300, 300), color="white")
        self.draw = ImageDraw.Draw(self.image)
        if os.path.exists("MACHINELEARNING/DRAW_PRED/image.jpg"):
            os.remove("MACHINELEARNING/DRAW_PRED/image.jpg")
        else:
            pass
    
    def printpred(self):
        im = Image.open('MACHINELEARNING/DRAW_PRED/image.jpg').convert("L")
        im1=im.resize((28,28))
        im2 = ImageOps.invert(im1)
        # threshold = 20
        # finimg = im2.point(lambda p: 255 if p > threshold else p)
        # finimg.show()
        arr=np.array(im2.getdata()).reshape(1,28,28,1)
        num=predictans(arr)
        print(num)
        return num

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
    