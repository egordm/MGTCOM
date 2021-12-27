# Research Questions

% The main goal of my thesis: to build a framework for community detection and representation in dynamic heterogeneous networks.
% 
% * This in order to analyse the communities within the set of datasets provided by Prof. Wang (**todo** cite work)
%   * Datasets are Dynamic, heterogeneous, Include content-based data
%   * Combination poses a problem
% * To the best of my knowledge: there are currently no algorithms that can do this
% * Direct comparison is therefore impossible

 The main goal of my thesis is to build a framework for community detection and representation in dynamic heterogeneous networks. 

This is, to enable dynamic community analysis on the datasets proposed in @wangPublicSentimentGovernmental2020. The data described is collected from the Twitter social platform and is dynamic, heterogeneous and rich in contentual (unstructured text) data. To the best of our knowledge, there are currently no dynamic community detection algorithms that can handle this data without relaxing its rich data representation (data loss).

As there are no alike algorithms, direct comparison is not possible. To both validate merit of our methods as well as the quality of the results, we spread our research over four research questions.



[Research Question 1]{#thm:rq1}

: *Does addition of meta-topological and/or content-based information improve quality of detected communities?*

% * Dataset rich in extra data: 
%   * meta-topological
%   * content based
% * Previous approach treat all nodes alike 
%   * ignoring most important structural features (node types)
%   * types such as hashtags can also be represented as nodes solving many issues which require topic modelling
% * Improvements in natural text analysis allow for representation of unstructured text
%   * Would this data improve quality of embeddings



[Research Question 2]{#thm:rq2}

: *Does usage of graph representation function learning techniques improve scale of CD beyond current state of the art?*

% * In last few years new graph representation approaches were introduced
%   * Instead of operating on the whole networks
%   * They sample network using random walks or convolutions
% * As previous methods for CD ignored them (used spectral methods)
%   * Causing scalability issues posed by spectral methods
%   * It is important to test these approaches as they may yield performance improvements
% * Previous approaches use spectral methods limiting them to one network per snapshot
% * Instead learning representation function
%   * would allow limit computational complexity
%   * allow for parameter sharing across timesteps
%   * Allow for streaming graphs



[Research Question 3]{#thm:rq3}

: *Does optimization for temporal smoothness provide better quality communities?*

% * 



[Research Question 4]{#thm:rq4}

: *Does making temporal features implicit in node representations provide better quality communities as opposed to making them explicit?*

% * Previous approaches either
%   * Learn the temporal component implicitly in node representations, causing embedding be temporally aware
%   * Sepaerate the temporal aspect explicitly by defining 


