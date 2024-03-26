#!/usr/bin/env python
# coding: utf-8

import os
import matplotlib.pyplot as plt
import cv2
from matplotlib.widgets import RectangleSelector

#global constants
img = None
tl_list = []
br_list = []
object_list = []
counter = 0

#constants
image_folder = 'data/hymenoptera_data/train/'
save_dir = 'data/hymenoptera_data_cropped/train/'
obj = 'ants'

def line_select_callback(clk,rls):
    #print(clk.xdata,clk.ydata)
    global tl_list
    global br_list
    tl_list.append((int(clk.xdata),int(clk.ydata)))
    br_list.append((int(rls.xdata), int(rls.ydata)))
    object_list.append(obj)

def onkeypress(event):
    global object_list
    global tl_list
    global br_list
    global img
    global counter
    if event.key == 'q':
        print(img.name, tl_list, br_list)
        crop_image(image, img, tl_list, br_list)
        tl_list = []
        br_list = []
        object_list = []
        img = None
        plt.close()

def crop_image(image, img, tl_list, br_list):
    i=1
    for i in range(len(tl_list)):

        x1, y1 = tl_list[i]
        x2, y2 = br_list[i]
        #print(x1, y1, x2, y2)
        cropped_img = image[y1:y2, x1:x2]
        #cv2.imshow("cropped", cropped_img)
        save_path = "{}/{}".format(save_dir,(str(i)+"_"+img.name))
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        status = cv2.imwrite(save_path, cropped_img)
        print("-->{} writing status: {}.".format(img.name, status))
        i = i+1

def toggle_selector(event):
    toggle_selector.RS.set_active(True)

if __name__ == '__main__':
    for n, folder in enumerate(os.scandir(image_folder)):
        print("->Cropping images in {}".format(folder.name))
        try:
            save_dir = "{}/{}/{}".format(image_folder, folder.name, folder.name)
            os.makedirs(save_dir)
        except FileExistsError:
            print("Folder exists.")

        for m, image_file in enumerate(os.scandir(folder)):
            img = image_file

            fig, ax = plt.subplots(1)

            image = cv2.imread(image_file.path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            ax.imshow(image)

            toggle_selector.RS = RectangleSelector(
                ax, line_select_callback, useblit=True,
                button=[1], minspanx=5, minspany=5,
                spancoords='pixels', interactive=True
            )

            bbox = plt.connect('key_press_event', toggle_selector)
            key = plt.connect('key_press_event', onkeypress)
            plt.show()
        print("->End Cropping images in {}-++++++++++++++++++++++++++++++++++++++++++++++++++++++")