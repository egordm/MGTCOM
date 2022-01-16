# Literature Review

The problem of dynamic community detection was noticed quite early on within the SNA community and a considerable amount of research has been made in order to provide a comprehensive analysis [@tamimiLiteratureSurveyDynamic2015; @rossettiCommunityDiscoveryDynamic2018; @dakicheTrackingCommunityEvolution2019]. While the said research was mostly focused on the discovery of communities using topologically-based features and node connectivity, the covered methods did research the limitations and challenges posed by a temporal context.

In recent years, significant developments have been made in the space of deep learning. Mainly in the development of new deep learning methods capable of learning graph-structured data [@bronsteinGeometricDeepLearning2017; @hamiltonRepresentationLearningGraphs2018; @kipfSemiSupervisedClassificationGraph2017] which is fundamental for SNA. Because of this, various problems within the field have been revisited, including community detection problems. The approaches have been expanded by incorporation of more complex features, solving the issues concerning multi-modality, and the introduction of unsupervised learning.

Despite this resurgence, the DCD problem has received little attention. Though a few efforts have been made to incorporate the deep learning methods by introducing content-based similarity [@faniUserCommunityDetection2020; @cazabetUsingDynamicCommunity2012; @huangInformationFusionOriented2022], the definition of unified constraints for end-to-end learning [@maCommunityawareDynamicNetwork2020; @wangEvolutionaryAutoencoderDynamic2020; @cavallariLearningCommunityEmbedding2017; @jiaCommunityGANCommunityDetection2019], and usage of graph representation-based CD algorithms [@wangVehicleTrajectoryClustering2020; @limBlackHoleRobustCommunity2016] within a temporal context, the current state-of-the-art leaves room for improvement.

% 

We structure the literature as follows: first, we describe the various interpretations of the Community Structure in +@community-structures. Next, we explore various approaches and techniques related to Graph Representation Learning in +@graph-representation-learning. Then, we provide an overview of the current state-of-the-art approaches for Community Detection and Dynamic Community Detection tasks in +@link-based-approaches and +@representation-based-approaches. Finally, we discuss the ways to evaluate the said algorithms in +@evaluation and the datasets available in +@datasets.


