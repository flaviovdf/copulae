# 2-Cats

![](2cats.webp)

**2d Copula Approximating Transforms** (2-Cats). 2-Cats is a neural network
approach to learn 2-D Copulas. It relies on monotonic transforms of the input data from the
**I** (\[0, 1\]) domain to the **R**eal (\[-∞, ∞\]) domain. Input data consists of the
Cumulative Distribution Functions (CDFs) values for 2-D marginals (e.g., x and y).

After this data is mapped to a vector of two numbers each on the real domain, the
marginals are copulated using any Bivariate Cumulative Probability Distribution.
