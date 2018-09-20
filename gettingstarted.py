import torch
import numpy as np

# creates an empty tensor with dimensions 5x3
x = torch.empty(5,3)
print(x)

y = torch.rand(5,3)
print(y)

z = torch.zeros(5,3, dtype=torch.long)
print(z)

a = torch.tensor([[5.5,3],[4.4, 3.3]])
print(a.size())

c = torch.randn_like(x, dtype=torch.float)
print(c)


x = x.new_ones(5,3,dtype=torch.float)
print(x)

y = torch.rand(5,3)

print(x+y)


# Output a tensor with x+y
result = torch.empty(5,3)
torch.add(x,y,out=result)
print(result)


y.add_(x)
print(y)

x.copy_(y)
print(x)


z = x.view(15)
print(z)



d = torch.ones(5)
print(d)

e = d.numpy()
print(e)


d.add_(1)

print(d)
print(e)


a = np.ones(5)
b = torch.from_numpy(a)

np.add(a, 1, out=a)
print(a)
print(b)


# Check if cuda is available
print(torch.cuda.is_available())


if torch.cuda.is_available():
	device = torch.device("cuda")
	y = torch.ones_like(x, device=device)
	x = x.to(device)

	g = x+y
	print(g)
	print(g.to("cpu", torch.double))