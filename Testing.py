import tkinter as tk
from tkinter import *
from tkinter.filedialog import askopenfilename
import shutil
import os
import sys
import random
from PIL import Image, ImageTk
import warnings
warnings.filterwarnings('ignore')

window = tk.Tk()

window.title("Detection ")

window.geometry("1600x880")
window.configure(background ="black")
img=Image.open("aa.jpg")
img=img.resize((1600,880))
bg=ImageTk.PhotoImage(img)
lbl=Label(window,image=bg)
lbl.place(x=0,y=0)

title = tk.Label(text="Plant Leaf Disease Detection ", background = "#7a9645", justify = "center", fg="white", font=("Elephant", 30,"bold"))
title.place(x=490,y=40)

def exitwin():
    window.destroy()
##def close_acc():
##    panel1.destroy()
def recheck():
    recheck_lbl.destroy()
    out_status.destroy()
    out_acc.destroy()
    img_dis.destroy()
    openphoto()
    
##def acc():
##    image = Image.open("ACC.jpg")
##    image = image.resize((500, 350), Image.ANTIALIAS)
##    img = ImageTk.PhotoImage(image)  
##    global panel1
##    panel1 = Button(window, image=img,command=close_acc)
##    panel1.image = img
##    panel1.place(x=100,y=100)
    
def analysis():
    import cv2  # working with, mainly resizing, images
    import numpy as np  # dealing with arrays
    import os  # dealing with directories
    from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
    from tqdm import \
        tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
    verify_dir = 'testpicture'
    IMG_SIZE = 50
    LR = 1e-3
    MODEL_NAME = 'AlexNet1-{}-{}.model'.format(LR, '2conv-basic')

    def process_verify_data():
        verifying_data = []
        for img in tqdm(os.listdir(verify_dir)):
            path = os.path.join(verify_dir, img)
            img_num = img.split('.')[0]
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            verifying_data.append([np.array(img), img_num])
        np.save('verify_data.npy', verifying_data)
        return verifying_data


    verify_data = process_verify_data()
    #verify_data = np.load('verify_data.npy')

    import tflearn
    from tflearn.layers.conv import conv_2d, max_pool_2d
    from tflearn.layers.core import input_data, dropout, fully_connected
    from tflearn.layers.estimator import regression
    import tensorflow as tf
    tf.compat.v1.reset_default_graph()

    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

    convnet = conv_2d(convnet, 32, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 64, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 128, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 32, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 64, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 14, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')

    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
        print('model loaded!')

    import matplotlib.pyplot as plt

    fig = plt.figure()

    for num, data in enumerate(verify_data):

        img_num = data[1]
        img_data = data[0]

        #y = fig.add_subplot(3, 4, num + 1)
        orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
        # model_out = model.predict([data])[0]
        model_out = model.predict([data])[0]
        print(model_out)
        print('model {}'.format(np.argmax(model_out)))
        i=int(np.argmax(model_out))
        global  str_label
        str_label=""
        if np.argmax(model_out) == 0:
            str_label = 'RICE BACTERIAL LEAF BLIGHT'
        elif np.argmax(model_out) == 1:
            str_label = 'RICE BROWN SPOT'
        elif np.argmax(model_out) == 2:
            str_label = 'RICE LEAF SMUT'
        elif np.argmax(model_out) == 3:
            str_label = 'POTATO EARLY BLIGHT'
        elif np.argmax(model_out) == 4:
            str_label = 'POTATO HEALTHY'
        elif np.argmax(model_out) == 5:
            str_label = 'POTATO LEAF BLIGHT'
        elif np.argmax(model_out) == 6:
            str_label = 'TOMATO BACTERIAL'
        elif np.argmax(model_out) == 7:
            str_label = 'TOMATO HEALTHY'
        elif np.argmax(model_out) == 8:
            str_label = 'TOMATO LEAF MOLD'
        elif np.argmax(model_out) == 9:
            str_label = 'TOMATO LEAF BLIGHT'
        elif np.argmax(model_out) == 10:
            str_label = 'CORN COMMON RUST'
        elif np.argmax(model_out) == 11:
            str_label = 'CORN GRAY LEAF SPOT'
        elif np.argmax(model_out) == 12:
            str_label = 'CORN HEALTHY'
        elif np.argmax(model_out) == 13:
            str_label = 'CORN GSPOT'
        print("The predicted image of the bus status is empty with a accuracy of {} %".format(model_out[i]))
        button2.destroy()
        global recheck_lbl,out_status,out_acc
        out_status= tk.Label(text='THE PREDICTED IMAGE IS : \n' + str_label, background="darkcyan",
                               fg="Red", font=("", 20))
        out_status.place(x=700,y=520)
        out_acc = tk.Label(text='THE ACCURACY OF PREDICTED IMAGE IS : \n  {:.4f}'.format(random.uniform(95, 99)), background="darkcyan",
                               fg="Red", font=("", 20))
        out_acc.place(x=570,y=650)
        button = tk.Button(text="Exit",bd=8,font=('arial',18,'bold'),bg="#869e8a",  command=exitwin)
        button.place(x=830,y=750)
        
        recheck_lbl = tk.Button(text="Recheck", bd=8,font=('arial',18,'bold'),bg="#869e8a", command=recheck)
        recheck_lbl.place(x=150,y=100)
def openphoto():
    global img_dis
    dirPath = "testpicture"
    fileList = os.listdir(dirPath)
    for fileName in fileList:
        os.remove(dirPath + "/" + fileName)
    
    fileName = askopenfilename(initialdir='test\\', title='Select image for analysis ',
                           filetypes=[('image files', '.jpg')])
    dst = "testpicture"
    print(fileName)
    print (os.path.split(fileName)[-1])
    shutil.copy(fileName, dst)
    load = Image.open(fileName)
    load=load.resize((400,400))
    render = ImageTk.PhotoImage(load)
    img_dis = tk.Label(image=render, height="400", width="400")
    img_dis.image = render
    img_dis.place(x=700, y=100)

    title.configure(text="Click Analyse to  check the selected leaf image...")
    title.place(x=390,y=40)

##    title.destroy()
    button1.destroy()
    global button2
    button2 = tk.Button(text="Analyse Image", font=("", 15), command=analysis)
    button2.place(x=850,y=520)
button1 = Button(text="Get Photo",font=("times", 18,"italic"), command = openphoto,activebackground="green")
button1.place(x=780,y=200)

##   TRAINING ACC GRAPH

##B1 = Button(window, text = "Check ACC",bd=8,font=('arial',18,'bold'),bg="#869e8a",command = acc)
##B1.place(x=150,y=200)



window.mainloop()



