# DeepACE

*qxdu edited on May 11, 2026*

The code for computational implementation of "Anchor-based ensemble learning for assessing regulatory function of DNA sequences".

# Introduction

DeepACE (Deep Anchor-based Cis-regulatory Evaluation) is a deep learning–based framework that reinterprets 16 functional genomic models as complementary, semantically meaningful encodings of regulatory activity. DeepACE reveals a consistent geometric structure in which non-functional sequences collapse toward a compact regime, whereas functional sequences remain broadly distributed. Motivated by this asymmetry, DeepACE quantifies regulatory function as the distance from non-functional anchors, yielding a continuous, model-invariant metric without task-specific supervision. Consequently, DeepACE accurately captures the effects of sequence variation, distinguishes disease-associated variants, and eliminates non-functional synthetic candidates up to 16-fold more efficiently in sequence design tasks, while achieving leading performance across diverse benchmarking datasets and revealing interpretable functional directions within the regulatory landscape.

![Figure 1](./Figs/F01_deepace_diagram/DeepACE_Fig1.png)

**Figure 1.** Figure 1: Overview of the DeepACE framework 

(a)	Schematic of the DeepACE approach. Input consists of raw DNA sequences, which are first transformed into functional representations by multiple functional genomics models (I). These representations are then integrated via ensemble modeling and projected through Principal Component Analysis (PCA) into a unified representation space (i.e., URS) (II). Finally, distances to randomly sampled non-functional anchor sequences are computed to quantify regulatory function in a model-invariant manner (III). DeepACE ultimately assigns each input sequence a distance-based score reflecting its regulatory activity.

(b)	Uniform Manifold Approximation and Projection (UMAP) of feature distributions for the same set of sequences across 16 predictive models (total n = 43,275), showing that each model encodes regulatory information in a distinct, model-specific manner. Colors correspond to different models as indicated in the legend.

(c)	Heatmap of cross-model correlations for predicted changes in K562 CAGE signal at the center position following 5-kb tile perturbations. Correlations were computed across predictive models based on the predicted center-position activity changes induced by each perturbation. Labels denote the perturbed tile’s offset relative to the sequence center (negative/left, 0/center, positive/right). The heatmap highlights the degree of agreement among models in capturing position-dependent regulatory effects across long-range perturbations.

(d)	Intra-class distances between functional and non-functional sequences (1:1 ratio) across 10 datasets. Top: DeepACE representation space, where distances are computed as Euclidean distances in URS. Bottom: original sequence space, where distances are computed using Levenshtein distance. Fold-change indicates how much larger the intra-class distance of functional sequences is relative to that of non-functional sequences.

