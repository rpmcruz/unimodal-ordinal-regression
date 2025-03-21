# Unimodal Distributions for Ordinal Regression

Code for Paper by Jaime S. Cardoso, Ricardo Cruz, and Tomé Albuquerque.

**This paper has been published in IEEE Transactions on Artificial Intelligence: [10.1109/TAI.2025.3549740](https://ieeexplore.ieee.org/document/10918699).**

**Abstract:** In many real-world prediction tasks, the class labels contain information about the relative order between the labels that is not captured by commonly used loss functions such as multicategory cross-entropy. In ordinal regression, many works have incorporated ordinality into models and loss functions by promoting unimodality of the probability output. However, current approaches are based on heuristics, particularly non-parametric ones, which are still insufficiently explored in the literature. We analyze the set of unimodal distributions in the probability simplex, establishing fundamental properties and giving new perspectives to understand the ordinal regression problem. Two contributions are then proposed to incorporate the preference for unimodal distributions into the predictive model: 1) UnimodalNet, a new architecture that by construction ensures the output is a unimodal distribution, and 2) Wasserstein Regularization, a new loss term that relies on the notion of projection in a set to promote unimodality. Experiments show that the new architecture achieves top-3 performance, while the proposed new loss term is very competitive while maintaining high unimodality.
