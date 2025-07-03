
## Introduction

This project addresses the challenges of current mood-detection applications 
by developing a web-based system that integrates active and passive data collection to enhance 
the accuracy of bipolar disorder symptom detection.

## Background
To mitigate the risk of experiencing bipolar disorder, 
our online mood tracking project could enhance self-monitoring of users’ mental health. 

## Objectives
Enhance Bipolar Disorder Symptom Detection by integreting machine learning for emotion recognition 
Empower Users with Early Detection and Self-Monitoring by daily mood logging and report of mood analysis 
Implementing emotion recognition data to measure the mood swing and cycle

## Methodology
The model is constructed using a pre-trained VGG-16 base with frozen layers, 
processing 224x224x3 Mel spectrogram inputs, followed by an average pooling layer, a 128-unit ReLU dense layer. I also added 50% dropout layers to address high variance
and improve our accuracy

## Conclusion
I have explored several models for this detected mood system including pre-trained networks based on ResNet50 and VGG16. 
I believe I have preform a satisfactory performance application on tracking on the mood and emotion status.
The future extensions of this work could integrate Psychological research to further adapt our models to the real world


