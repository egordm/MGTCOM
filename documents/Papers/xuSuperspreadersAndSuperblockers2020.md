---
type: paper
title: "Superspreaders and superblockers based community evolution tracking in dynamic social networks"
author: "Xu et al."
creator: "Egor Dmitriev (6100120)"

---

# Superspreaders and superblockers based community evolution tracking in dynamic social networks - Xu et al.

## Goals

- Introduce a novel two-stage method that circumvents both of these problems simultaneously. Firstly, we propose an error accumulation sensitive (EAS) incremental community detection method for dynamic social networks.
  - A dynamic network snapshot is totally re-partitioned once the error accumulation degree of incremental clustering exceeds a pre-defined threshold

## Preliminaries

- Discovering new communities and tracking their evolution in social networks can provide valuable insights into the
  networks’ internal structure and their underlying organizational principles, with wide-ranging applications such as **forecasting emerging market trends in online retail networks** [4], **characterizing functions of unknown proteins** and **disease pathways in metabolic interaction networks** [5], **real-time partitioning of**
  **web-pages with different topics** [6], or **predicting the emergence and lifetimes of overlapping communities** in online social networks [3,7–9]

## Challenges

- Incrementally detecting network communities may result in partition errors such that continuous error accumulation
- Core-node-based methods have been widely employed; however, they do not distinguish between the heterogeneous contributions
- Since in a dynamic network a given community is likely to co-exist in several consecutive snapshots (and as such needs to be computed only once), methods based on static community detection will **inevitably suffer from efficiency shocks due to repeated calculation** of the same communities and the associated longer running-time consumption.
  - Due to these limitations, researchers have proposed an **evolutionary clustering method**

## Previous Work / Citations

- Core nodes with their various attributes can have very distinct contributions to different types of evolutionary events, i.e., a single type of core nodes cannot precisely reveal all kinds of critical evolution events
  - Identifying influential core nodes: Superspreader and superblocker nodes
- **This Work:** 
  - We **propose an error accumulation sensitive (EAS) algorithm** for dynamic community detection which effectively optimizes the incremental community detection performance.
  - We define a BATC (Balancer between estimated partition Accuracy and Time Cost) metric to obt**ain an appropriate error accumulation threshold** for the EAS algorithm
  - We **utilize superspreader and superblocker nodes** to identify critical evolution events

## Definitions

* Community structure: i.e., groups of nodes with a higher within-group connection density and a sparser connectivity across different groups 

## Outline / Structure

- A dynamic network snapshot is totally re-partitioned once the error accumulation degree of incremental clustering exceeds a pre-defined threshold

## Evaluation

- ...

## Code

- ...

## Resources

- ...
