import tkinter
import tkinter.messagebox
import tkinter.filedialog
import webbrowser
from PIL import Image, ImageTk
import PIL.Image
from tkinter import *
from tkinter.ttk import *
from tensorflow.keras.models import load_model
from eval_model import generate_desc, extract_features
from pickle import load
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

gui = Tk(className='Image-Captioning')
gui.geometry("650x650")


f_path = ""

lbl1 = Label(gui, text="1. Please select an image first.",anchor='w',font=("Courier", 12))
lbl2 = Label(gui, text="2. Then press the generate caption button. After this you will hear the audio " ,anchor='w',font=("Courier", 12))
lbl3 = Label(gui, text="   caption for the selected image and the text will be shown in the pop-up. ",anchor='w',font=("Courier", 12))
lbl4 = Label(gui, text="3. Now according to your choice you can google search for the caption or ",anchor='w',font=("Courier", 12))
lbl5 = Label(gui, text="   image search for similar images.", anchor='w',font=("Courier", 12))
lbl6 = Label(gui, text="                                                     ")

lbl1.pack(side = "top",fill='both')
lbl2.pack(side = "top",fill='both')
lbl3.pack(side = "top",fill='both')
lbl4.pack(side = "top",fill='both')
lbl5.pack(side = "top",fill='both')
lbl6.pack(side = "top",fill='both')

def google_search(final_cap):
   url = "https://www.google.com.tr/search?q={}".format(final_cap)
   webbrowser.open_new_tab(url)


def image_search(final_cap):
   url = "https://www.google.com.tr/images?q={}".format(final_cap)
   webbrowser.open_new_tab(url)


def select_image():
   f = tkinter.filedialog.askopenfilename(
      parent=gui, initialdir='..\imgs',
      title='Choose file',
      filetypes=[('jpg images', '.jpg')]
   )
   
   lbl1.destroy()
   lbl2.destroy()
   lbl3.destroy()
   lbl4.destroy()
   lbl5.destroy()
   lbl6.destroy()

   global f_path
   f_path= f_path+str(f)
   im = PIL.Image.open(f)
   tkimage = ImageTk.PhotoImage(im)
   myvar = Label(gui, image=tkimage)
   myvar.image = tkimage
   myvar.pack()
   # print(f)

b1 = tkinter.Button(gui, text='Select Image', command=select_image, bg = 'white')
b1.pack()

cap_gen=""
# print(f_path)

def generate_cap(f_path):
   if f_path == "":
      tkinter.messagebox.showinfo("ERROR MESSAGE","You have not selected any image !")
      return
   global cap_gen
   tokenizer = load(open('../models/tokenizer.pkl', 'rb'))
   index_word = load(open('../models/index_word.pkl', 'rb'))
   # pre-define the max sequence length (from training)
   tokenizer.oov_token = None
   max_length = 34
   # load the model
   filename = '../models/wholeModel_vgg19.h5'
   model = load_model(filename)

   photo = extract_features(f_path)
      # generate description
   captions = generate_desc(model, tokenizer, photo, index_word, max_length)

   final_cap = ' '.join(captions[4][0].split()[1:-1])
   cap_gen = final_cap
   language = 'en'
   
   tkinter.messagebox.showinfo("*Caption*", final_cap)

   import pyttsx3

    # initialisation2
   engine = pyttsx3.init()
   engine.save_to_file(final_cap, r'C:\Users\DELL\Documents\komal_new\imgs\Predicted_audio.mp3')

   engine.say(r'C:\Users\DELL\Documents\komal_new\imgs\Predicted_audio.mp3')
   
   B1 = tkinter.Button(gui, text="Enter for Google search", command=lambda: google_search(cap_gen), bg = 'white')
   B2 = tkinter.Button(gui, text="Enter for Image search", command=lambda: image_search(cap_gen), bg = 'white')

   B1.pack()
   B2.pack()

B = tkinter.Button(gui, text ="Generate Caption", command = lambda : generate_cap(f_path), bg = 'white')
B.pack()

gui.mainloop()
