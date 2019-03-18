# processing
import cv2
import numpy as np
# gui
from tkinter import filedialog
import tkinter as tk
# conversion
from PIL import Image
from PIL import ImageTk

def choose_and_open_video():
    path = filedialog.askopenfilename(initialdir = '/', title = 'Select video to examine', filetypes = (('MP4', '*.mp4'), ('AVI', '*.avi')))
    return cv2.VideoCapture(path)

def numpy_to_tk(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    return ImageTk.PhotoImage(image)

root = tk.Tk()
root.title('Mouth Motion Analysis')

menubar = tk.Menu(root)
menubar_file = tk.Menu(menubar, tearoff = 0)
menubar_file.add_command(label = 'Open', command = choose_and_open_video)
menubar_file.add_command(label = 'Exit', command = root.destroy)
menubar.add_cascade(label = 'File', menu = menubar_file)

root.config(menu = menubar)
# button = tk.Button(m, text='Quit', width=25, command = m.destroy)
# button.pack()
#
# canvas = tk.Canvas(m, width = 40, height = 60)
# canvas.pack()
#
# list = tk.Listbox(m)
# list.insert(1, 'Python')
# list.pack()

# m.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
# print(m.filename)
root.mainloop()
