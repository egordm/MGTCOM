---
type: paper
title: "Community Discovery in Dynamic Networks: A Survey"
author: "Rossetti et al"
creator: "Egor Dmitriev (6100120)"

---

# Community Discovery in Dynamic Networks: A Survey - Rossetti et al 



## Goals

- ...

## Preliminaries

- New field of investigation that emerged in the past decade: dynamic network analysis
  - Dynamic community discovery (DCD)
    - tracking of local topologies and of their mutations
- **Not relevant to characterize the nature of temporal networks**
  - Transformed into an interval graph by extending edges of an infinitesimal amount
- Two types of **temporal network**, **interaction networks** and relation networks

## Challenges

- As illustrated by the famous paradox of Theseus, deciding if an **element composed of several entities** at a given instant is the same or not as another one composed of some—or even none—of such entities at a later point in time is **necessarily arbitrary and cannot be answered unambiguously**
- Main issues encountered by dynamic community detection approaches is the **instability of solutions**
- Another issue is that the choice between one partition and another is somewhat arbitrary

## Previous Work / Citations

- ...
- **This Work:** ...

## Definitions

* Community: A community in a complex network is a set of entities that share some closely correlated sets of actions with the other entities of the community. We consider a direct connection as a particular and very important kind of action.
* Dynamic Community Discovery: Given a dynamic network DG, a dynamic community DC is defined as a set of distinct (node, periods) pairs
* Operations: 
  * Birth: The first appearance of a new community composed of any number of nodes.
  * Death: The vanishing of a community: all nodes belonging to the vanished community lose
    this membership.
  * Growth: New nodes increase the size of a community.
  * Contraction: Some nodes are rejected by a community, thus reducing its size.
  * Merge: Two communities or more merge into a single one.
  * Split: A community, as consequence of node/edge vanishing, splits into two or more components.
  * Continue: A community remains unchanged.
  * Resurgence: A community vanishes for a period, then comes back without perturbations as if it has never stopped existing. This event can be seen as a fake death-birth pair involving the same node set over a lagged time period (e.g., seasonal behaviors).

![Screenshot_20211116_225452](rossettiCommunityDiscoveryDynamic2018.assets/Screenshot_20211116_225452.png)



## Outline / Structure

- 

## Evaluation

- ...

## Code

- ...

## Resources

- ...