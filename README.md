# Machine Learning

This sample code is for the just the identification of people in family.
You may use this method to add some more security layer at your residence.

Problem Statement:
    To identify the individuals in home

Data Set Available:
    3 Classes available for training of network.
    1. Myself
    2. My dear wife
    3. None (Image which doesn't contain either of us)

Flow:
    Using opencv, created the data set. Captured the images of both of us and named them like 'Siddhesh_123.jpg' and 'None_123.jpg'.
Used one hot vector for identification of the people.
personDict = {'Siddhesh':[1., 0., 0.],
              'Wife': [0., 1., 0.],
              'None': [0., 0., 1.]}

Converted the image to vec of size 1, 625000 (image size taken was 250 * 250)
Fed this to the network and saved the model after n epochs.

I could get the accuracy upto 75% from 30 epochs and ~3400 training examples.

Using the saved model, inferenced the live feed from usb camera.
