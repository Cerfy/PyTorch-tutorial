import torch

# requires_grad = True allows tracking of computation
x = torch.ones(2,2, requires_grad = True)


y = x + 2

z = y * y * 3
out = z.mean()

# Scalar output
print(out)

a = torch.zeros(2,2)
a.requires_grad_(True)
print(a)

# References a function that has created the tensor y.
print(y.grad_fn)


# Computes the gradient of x
out.backward()
print(x.grad)


p = torch.randn(3, requires_grad = True)
n = p * 2

# Computing the L2 Norm by calling the below method
print(n.data.norm())

while n.data.norm() < 1000:
	n = n * 2

print(n.grad_fn)

gradient_tensor = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)

# Pass in values of which the derivative will be computed.
n.backward(gradient_tensor)

print(p.grad)
