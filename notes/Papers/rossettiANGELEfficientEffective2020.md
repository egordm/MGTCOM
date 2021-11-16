---
title: "ANGEL: efficient, and effective, node-centric community discovery in static and dynamic networks"
author: "Rossetti et al"
creator: "Egor Dmitriev (6100120)"

---

# ANGEL: efficient, and effective, node-centric community discovery in static and dynamic networks - Rossetti et al 

Our approach is primarily designed for social networks analysis and belongs to a well-known subfamily of Community Discovery approaches often identified by the keywords bottom-up and node-centric

## Goals

- we propose ANGEL , an algorithm that aims to lower the computational complexity of previous solutions while ensuring the identification of high-quality overlapping partitions

## Preliminaries

- ...

## Challenges

- complex networks researchers agree that it is not possible to provide a single and unique formalization that covers all the possible characteristics a community partition may satisfy

## Previous Work / Citations

- (Coscia et al. 2012): where the authors propose DEMON an approach whose main goal was to identify local communities by capturing individual nodes perspectives on their neighbourhoods and using them to build mesoscale ones
- 
- **This Work:**
  - Introduces a Label Propagation algorithm
    - Least complex kind of algorithm
    - Gives good quality results
  - In contrast to DEMON it focuses on lowering the time complexity while at the same time increasing the partition quality
  - Properties:
    - It produces a deterministic output 
    - Allows for a parallel implementation

## Definitions

* During each iteration, the label of v is updated to the majority label of its neighbours. As the labels propagate, densely connected groups of nodes quickly reach a consensus on a unique label

## Outline / Structure

- ...

## Evaluation

- ...

## Code

- ...

## Resources

- ...