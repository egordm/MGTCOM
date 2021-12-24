## Community Structures

The goal of this section is to introduce fundamental structures for the Dynamic Community Detection task. We do this by combining various definitions used in the relevant literature as well as establishing the purpose for these structures, before proceeding into approaches for detecting communities in the following sections.



% Goals:
% 
% * Give a general definition of important structures



### Communities

Communities in real-world networks can be of different kinds: disjoint (students belonging to different educational institutions), overlapping (person having membership in different social groups) and hierarchical (components of a car). One of the main reasons behind the complexity of CD is that there is not one unique definition what a community actually is.

The *link-based* (also referred to as classic) community detection methods intuitively describe communities as groups of nodes within a graph, such that the intra-group connections are denser than the inter-group ones. This definition is primarily based on the *homophily* principle, which refers to the assumption that similar individuals are those that are densely connected together. Therefore, these kind of methods look for sub-graph structures such as cliques and components that identify connectedness within the graph structure to represent the communities.

Unfortunately, in most cases link-based methods fall short to identify communities of similar individuals. This is mainly due to two facts: (i) many similar individuals in a social network are not explicitly connected together, (ii) an explicit connection does not necessarily indicate similarity, but may explained by sociological processes such as conformity, friendship or kinship [@diehlRelationshipIdentificationSocial2007; @faniUserCommunityDetection2020].

A more general definition is introduced in [@cosciaClassificationCommunityDiscovery2011] to create an underlying concept generalizing all variants found in the literature (+@thm:community). In link-based methods, a direct connection is considered as a particular and very important kind of action, while newer methods also consider content or interest overlap.

[Community]{#thm:community}

: A community in a complex network is a set of entities that share some closely correlated sets of actions with the other entities of the community.



### Dynamic Communities

Similar to how communities can be found in static networks, dynamic communities extend this definition by utilizing the temporal dimension to define their life cycle/evolution over a dynamic network. A dynamic community is characterized by a collection of communities and a set of transformations on these communities over time.

This persistence of communities across time subjected to progressive changes is an important problem to tackle. Though, as noted by [@rossettiCommunityDiscoveryDynamic2018] the problem can be compared to the famous “the ship of Theseus” paradox. Because (verbatim), *deciding if an element composed of several entities at a given instant is the same or not as another one composed of some—or even none—of such entities at a later point in time is necessarily arbitrary and cannot be answered unambiguously*.

Most of the works agree on two atomic transformations on the communities, including node/edge appearance and vanishing. While some such as [@pallaQuantifyingSocialGroup2007; @asurEventbasedFrameworkCharacterizing2009; @cazabetUsingDynamicCommunity2012] define a more extensive set of transformations (also referred to as events) which may be more interesting for analytical purposes:

* Birth, when a new community emerges at a given time. 
* Death, when a community disappears. All nodes belonging to this community lose their membership.
* Growth, when a community acquires some new members (nodes).
* Contraction, when a community loses some of its members.
* Merging, when several communities merge to form a new community.
* Splitting, when a community is divided into several new ones.
* Resurgence, when a community disappears for a period and reappears.

These events/transformations are often not explicitly used during the definition and/or representation of dynamic communities. Nevertheless, most of the methods covered in the following sections do define a way in their algorithm to extract such events from the resulting data.

Finally, it is important to note that dynamic networks can differ in representation. They can be represented as either a time series of static networks (also referred to as snapshots) or as a real-time stream of edges (referred to as temporal networks). Within the global context of dynamic community detection, they can be seen as equivalent as the conversion between the two representations can be done in a lossless way. The latter, temporal networks are often used to handle incremental changes to the graph and are most commonly applied within real-time community detection settings.




