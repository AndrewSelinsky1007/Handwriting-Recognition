# Handwriting-Recognition CNN Model

This is a **Handwritting Recognition Engine** powered by a Convolution Neural Network model trained to recognize alphanumerical characters.

## Prerequisites
* **Python3.x**
* **A connected camera**
* **Tensorflow, Numpy, and OpenCV**

## Training the model
The model must be trained before use of the program. Run the script corresponding to the characters you want the program to recognize.
* `python trainLetters.py` 
* `python trainNumbers.py`

## Running the model
Once a model is successfully trained, run the corresponding script to the model you trained.
* `python runLetters.py`
* `python runNumbers.py`

You should see two windows appear, one of your raw camera stream, and one formatted for what the model is being passed.

## Usage and video input
You must present your handwritting correctly for the program to recognize.
* You must use a thick, dark marker on white paper.
* Hold the character close to the camera to fit the box.
* When recognizing letters, only capital letters are recognized.

## Digit vs. Letter recognition
* The program for recognizing digits will only handle one at a time.
* The program for recognizing letters can be used to build sentences in real time by holding a letter in the box on screen for 20 frames to add it to the string.

## Controls
* **Spaces** Press 'space' to add a space to the string.
* **Clear** Press 'c' to clear the current string.
* **Quit** Press 'q' to exit the program.
