# MRIScanClassification

This is a simple Scala program that uses mllib library to implement 
an ensemble of tree based classifiers.

Problem Background

Scientists are interested in turning high-resolution brain scans into a graph  
representing nerve connections indicated by the bright lines. 
Note that the original image is 3-dimensional, 
but for better intuition we show the 2-dimensional projection on the X-Y plane.
(Image is available with the wiki link).
The image is noisy, but axons are clearly visible. 

The scientists have put in a lot of effort, manually tracing the axons. 
Intuitively, the traced data looks like the image, 
where white indicates foreground and black indicates background. 
(In reality, there is also a 2-pixel wide “zone of uncertainty” between 
white lines and background. We do not show it separately to avoid clutter.)

Labeled Input Data
Using the manually traced image, we generated 
labeled data as follows:
• Select a pixel (i, j, k) in the image.
• Extract a neighborhood vector centered around (i, j, k).
• Extract the label of (i, j, k) from the trace.
  Here 1 indicates foreground; 0 indicates background.
• Save the record as the neighborhood vector, followed by the label.

Task
To improve the quality of algorithms that automatically trace these axons 
in an image, we want to classify each pixel as foreground (belongs to an axon) 
or background (does not belong to an axon).

Intuition:
Running through some statistics, around 99.5% of the given dataset for training
is background labels while 0.5% of the data only contains foreground. 
This is a highly imbalanced learning task. 
A simple, scalable approach, would be to use a deep RandomForest model
while giving weight to the classification label in the ratio of its frequency 
in the given dataset.
Since we are limited to creating a depth of only 30 for decision tree based models
using Spark, and since we are also unable to automatically weight the classification
labels during training as there is no such option provided by Spark, we decided to
implement an ensemble approach, where we used Gradient Boosted Trees with a depth of 2
to offset the limitations of Random Forest.

With such an ensemble of classifiers, we were able to beat the baseline.  

Platform
Scala is used to implement the machine learning model.
The model was trained on AWS EMR with 20 instances.