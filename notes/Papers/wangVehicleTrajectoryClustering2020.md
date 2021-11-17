---
title: "Vehicle Trajectory Clustering Based on Dynamic Representation Learning of Internet of Vehicles"
author: "Wang et al."
creator: "Egor Dmitriev (6100120)"

---

# Vehicle Trajectory Clustering Based on Dynamic Representation Learning of Internet of Vehicles - Wang et al. 



## Goals

- We propose to employ **network representation learning to achieve accurate vehicle trajectory clustering**
  - Specifically, we **first construct the k-nearest neighbor-based** internet of vehicles in a dynamic manner
  - We **learn the low-dimensional representations** of vehicles by performing dynamic network representation learning
  - Using the learned vehicle vectors, **vehicle trajectories are clustered**
- 

## Preliminaries

- Vehicle trajectory clustering aims to regroup similar vehicle trajectories together into different groups
  - Extract relevant information in order to, for instance, calculate the optimal path from one position to another, detect abnormal behavior, monitor the traffic flow in a city, and predict the next position of an object
  - The road networks of different city regions may be totally different
  - Vehicle may present totally different trajectories over different time periods of a day
  - Meanwhile, the patterns on weekdays and weekends may also different.

## Challenges

- As the location of vehicles is constantly changing, the vehicle social network is a dynamic network

## Previous Work / Citations

- ...
- **This Work:** ...

## Definitions

* …

## Outline / Structure

- To construct the dynamic vehicle network, we regard **vehicles as nodes in the network**, so we get the node
  set V . For every two nodes (vi and vj ) and in V , in order to **determine whether there is an edge** (eij ) between them, we **divide the region into many small squares with length** and width of 0.001◦ according to longitude and latitude
- We propose to **learn the embedding vectors of vehicles** by performing **dynamic network representation learning** on the previously constructed k-nearest neighbor-based vehicular network
- DynWalks:
  - Performs truncated random walks with length $l$ on each selected node for $r$ times
  - By using a silding window with length w + 1 + w to slide on each random walk sequence
  - Uses the Skip-Gram **Negative Sampling** (SGNS)
  - DynWalks only **performs random walks on selected nodes** and updates the embedding vectors of selected nodes
    - The **embedding vectors of other nodes remains unchanged**
    - Updated based on incremental updates in $t$
- Clustering:
  - K-means, K-medoids, GMM
  - Performed on each timestep
  - Loops through possible cluster counts :S

## Evaluation

- ...

## Code

- https://github.com/HansongN/dynamic_vehicle_network_clustering

## Resources

- ...