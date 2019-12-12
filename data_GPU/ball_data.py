#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 15:21:37 2019

@author: nrchilku
"""
import pygame
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

def main(width, height, samples, frames, save=False):
    
    X = []
    Y = []
    screen = pygame.display.set_mode((width, height))
    end = False       
    for sample_count in range(samples):
        radius = np.random.randint(2, 7)
        x = np.random.randint(radius + 5, width - radius - 5) # initial x-position
        y = np.random.randint(radius + 5, height - radius - 5) # initial y-position
        v_x = np.random.randint(-2, 2) # initial x-velocity
        v_y = np.random.randint(-2, 2) # initial y-velocity
        color = np.random.randint(50, 255, 3) # background is black so starting at 50
        X_frames = []
                            
        for frame_count in range(frames+5):                
            x += v_x
            y += v_y
            
            screen.fill((0, 0, 0))
            pygame.draw.circle(screen, color, (x,y), radius)
            X_frames.append(pygame.surfarray.array3d(screen))            
            # check for bounce
            if x <= radius or x >= width - radius:
                v_x *= -1
            if y <= radius or y >= height - radius:
                v_y *= -1  
            #clock.tick(30)
            pygame.display.flip()
        # append X and Y
        X.append(np.array(X_frames[:-5]))
        # Y
        grayscale = np.dot(np.array(X_frames)[5:], np.array([0.2125, 0.7154, 0.0721]))
        Y.append(grayscale.astype('uint8'))
                
    
    if save:
        np.save("X_5_" + str(width) + "_" + str(frames) + "_" + str(samples), np.array(X))
        np.save("Y_5_" + str(width) + "_" + str(frames) + "_" + str(samples), np.array(Y))
            
    #return np.array(X), np.array(Y)
    pygame.quit()
                    
def plot(pred, X, Y, n_plots):
    
    
    high = pred.shape[0]
    samples = np.random.randint(0, high, n_plots)
    count = 1
    
    for sample in samples:
                
        frame = np.random.randint(0, 15)    
           
        plt.subplot(n_plots, 3, count)
        plt.imshow(X[sample, frame, :, :, 0])        
        if count == 1:
            plt.title('Input')
        plt.subplot(n_plots, 3, count+1)
        plt.imshow(Y[sample, frame, :, :, 0])
        if count == 1:
            plt.title('Ground Truth')
        
        
        plt.subplot(n_plots, 3, count+2)
        plt.imshow(pred[sample, frame, :, :, 0])
        if count == 1:
            plt.title('Prediction')
         
        count += 3
        plt.show()
             
                
def gif(pred, name):
    fig = plt.figure()
       
    im = []
    for sample in range(pred.shape[0]):
        for frame in range(pred.shape[1]):
            im.append([plt.imshow(pred[sample, frame, :, :, 0])])
            
    animate = animation.ArtistAnimation(fig, im, blit=True, interval=20, repeat_delay=500)
    writer = PillowWriter(fps=20)
    animate.save(name+str('.gif'), writer=writer)
    plt.show()
               
            
import imageio
import glob

def c_pngs():
    arr = np.zeros((16, 40, 40, 3))
    j = 0
    for im_path in sorted(glob.glob('../stim_GPU/*.png')):
         im = imageio.imread(im_path)
         arr[j, :, :, :] = im
         j += 1
    return arr

def images(X, Y, n):
    w = 85
    h = 45*n
    fig=plt.figure(figsize=(w, h))
    columns = 2
    rows = n
    j = 1
    for i in range(1, columns*rows +1):
        
        s = np.random.randint(0, 100)
        f = np.random.randint(0, 16)
        fig.add_subplot(rows, columns, j)
        plt.imshow(X[s, f, :, :, :])
        
        fig.add_subplot(rows, columns, j+1)
        plt.imshow(Y[s, f, :, :])
        
        j+=2        
        
    plt.show()
            
