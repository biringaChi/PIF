# Predictive Frame Inference (PIF) Model

## License
PIF-Model is licensed under **Apache Software License, Version 2.0**.

## Description
In this project, we built an AI model that could infer a new frame between two existing sequential frames of a video.
We trained a GAN neural network framework using youtube footage to predict it’s own existing frames. The model which was created was able to infer new frames from high definition videos.

## Methodology
* Model
	*  Trained using GAN
	* Convolutional Neural Network Model with multiple input
	* Images are encoded and decoded
    
	<p align="center"><img width="100%" src="readme/GANs.png" alt="gan"/></p>
    
* Tools / Resources
    * PyTorch was used to build the models
    * Royalty-free videos was collected from Youtube

* Performance Measure
    * Pixels distance was used as a potential loss for the generator and metric to measure difference
    * BSE was used as the loss for the model
    
    <p align="center"><img width="40%" src="readme/sframes.png" alt="sframes"/></p>
    

* Dataset
	* Videos was split into subset of 3 continuous frames
	* The model will be trained to predict the current frame by using the previous and next frame in the video
	* The model was able to generate new frames from the data distribution
	* Stitching technique was implemented to build the videos
    
## Run Sample
```
python data_sample.py
```
<p align="center"><img width="100%" src="readme/sd1.png" alt="data_sample"/></p>
32 samples from dataset

* Scene Detection
    * Contiguous frame sequences were used by comparing the difference each scene to the next using.
    * The spikes indicate transitions or new scenes. These were used as markers to tell the model where not to infer new frames.
<p align="center"><img width="100%" src="readme/scenes.png" alt="scenes"/></p>
Detected scenes

## Results
![R1](readme/R1.png) ![R2](readme/R2.png) ![R2](readme/R3.png)
Infered Frames from GAN Model
