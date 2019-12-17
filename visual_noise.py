#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Demo: Using an arbitrary numpy array as a visual stimulus.

Also illustrates logging DEBUG level details to the console.
"""

from __future__ import division

from psychopy import visual, event, core, logging
import numpy

logging.console.setLevel(logging.DEBUG)

win = visual.Window([224, 224], allowGUI=False, color=(-1, -1, -1))

noiseTexture = numpy.random.rand(16, 16) * 2.0 - 1

patch = visual.GratingStim(win, tex=noiseTexture,
    size=(32, 32), units='pix', mask="circle",
    interpolate=True, autoLog=False, blendmode="avg", ori=0, pos=(-10, 15))
    
#patch = visual.NoiseStim(win, mask='circle', units='pix', pos=(0.0, 0.0), size=(128, 128), noiseType="binary", noiseElementSize=32, interpolate=True, blendmode='avg')

while not event.getKeys():
    patch.phase += (-1 / 300.0, 0)
    patch.pos += (0, -1)
    patch.draw()
    win.flip()
    win.getMovieFrame() 


win.close()
win.saveMovieFrames('0.png') # use win.saveMovieFrames('name.gif') to save it as a gif
core.quit()

# The contents of this file are in the public domain.
