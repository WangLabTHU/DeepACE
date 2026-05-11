# DeepACE

*qxdu edited on May 11, 2026*

The code for computational implementation of "Anchor-based ensemble learning for assessing regulatory function of DNA sequences".

# Introduction

DeepACE (Deep Anchor-based Cis-regulatory Evaluation) is a deep learning–based framework that reinterprets 16 functional genomic models as complementary, semantically meaningful encodings of regulatory activity. DeepACE reveals a consistent geometric structure in which non-functional sequences collapse toward a compact regime, whereas functional sequences remain broadly distributed. Motivated by this asymmetry, DeepACE quantifies regulatory function as the distance from non-functional anchors, yielding a continuous, model-invariant metric without task-specific supervision. Consequently, DeepACE accurately captures the effects of sequence variation, distinguishes disease-associated variants, and eliminates non-functional synthetic candidates up to 16-fold more efficiently in sequence design tasks, while achieving leading performance across diverse benchmarking datasets and revealing interpretable functional directions within the regulatory landscape.

![Figure 1](./Figures/github/Figure1.png)
**Figure 1.** Overview of the DeepSwitch framework, including sequence generation, multi-layered prediction, iterative multi-objective optimization and massively parallel reporter assays.
