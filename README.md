# 2-Cats

![](2cats.webp)

**2d Copula Approximating Transforms** (2-Cats). 2-Cats is a neural network
approach to learn 2-D Copulas. It relies on monotonic transforms of the input data from the
**I** (\[0, 1\]) domain to the **R**eal (\[-∞, ∞\]) domain. Input data consists of the
Cumulative Distribution Functions (CDFs) values for 2-D marginals (e.g., x and y).

After this data is mapped to a vector of two numbers each on the real domain, the
marginals are copulated using any Bivariate Cumulative Probability Distribution.

# Using the Model

```python


layer_widths = [128, 64, 32, 16]

losses = [
    (0.01, sq_error),
    (0.5, sq_error_partial),
    (0.1, copula_likelihood),
]
lr = 2e-3
n_iter = 2000

D = # some 2d dataset with dimensions on columns (2, n) shape. Transpose a (n, 2) data.
TrainingTensors = generate_copula_net_input(
    D=D,  # 2 cats receives transposed data for historic reasons
    bootstrap=False
)

# Symbolic differentitation
# nn_C is the Copula
# nn_dC is the first derivative
# nn_c  is the second derivative, pseudo-likelihood
# cop_state is a training state for Flax
# forward is the forward function to perform gradient descent
# grad is the derivative to perform gradient descent
nn_C, nn_dC, nn_c, cop_state, forward, grad = setup_training(
    model, TrainingTensors, losses, rescale=True
)

# Initialize Model
key, subkey = jax.random.split(key)
init_params = model.init(subkey, TrainingTensors.UV_batches[0])
del subkey

params = init_params
optimizer = optax.adam(lr)
opt_state = optimizer.init(params)

# Training Loop
for i in tqdm(range(n_iter)):
    grads, cop_state = grad(params, cop_state)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

```
