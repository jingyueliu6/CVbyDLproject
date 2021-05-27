# CVbyDLproject
## Introduction to project
The goal of this reproduction project is to reproduce some results of the paper **'Measuring economic activity from space: a case study using flying airplanes and COVID-19'**. The interesting thing about the paper is the combination of computer vision on satellite imagery for a wide range of spatial areas. The paper basically aims at producing a flying airplane detector for satellite imagery and uses this data to produce time series data of airport activity. The airport activity is, according to the authors, an indicator for economic activity. It is interesting to work with satellite image data as it holds a lot of potential for computer vision projects.

Another interesting aspect of the project is the use of a relatively small amount of annotated images. The authors chose this approach as it makes the overall method suitable for a fast implementation for projects where large annotated datasets do not exist. Although there is a small amount of annotated data, the overall amount of data is huge. This is due to the large file sizes produced by the satellite imaging systems. For us it will be a challenge dealing with these large datasets, therefore we are considering using only a portion of the testing and validation data as the paper uses.

## Sampling strategy
![image](https://user-images.githubusercontent.com/36470382/119887149-0d766680-bf34-11eb-874f-f626ba7eb47c.png)

The sampling strategy for training will be explained here. First, a list with annotations (containing 
