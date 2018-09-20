import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
	# Initialization of layers
	def __init__(self):
		super(Net, self).__init__()
		# Convolution layers 
		self.conv1 = nn.Conv2d(1,6,5) # 1 input channel, 6 output channels 5x5 Conv
		self.conv2 = nn.Conv2d(6,16,5) # 6 input channel, 16 output channels 5x5 Conv
		# Full connected layers
		self.fc1 = nn.Linear(16*5*5, 120) # 16 5x5 = 16*5*5 number of input parameters / Output 120 parameters
		self.fc2 = nn.Linear(120, 84) # 120 input parameters to 84 output parameters
		self.fc3 = nn.Linear(84, 10) # 84 input parameters to 10 output parameters


	# Specify the connections between layers
	def forward(self, x):
		# Max pooling over 2x2 window and non-linear function ReLu after first conv layer
		x = F.max_pool2d(F.relu(self.conv1(x)), 2)
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)
		x = x.view(-1, self.num_flat_features(x)) # -1 means the dimension is inferred from the other
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x


	def num_flat_features(self, x):
		size = x.size()[1:]
		num_features = 1
		for s in size:
			num_features *= s
		return num_features


if __name__ == '__main__':
	net = Net()
	# print(net)
	# params = list(net.parameters())
	# print(len(params))
	# print(params[0].size()) # Conv 1's Weights


	input = torch.randn(1,1,32,32)
	# out = net(input)
	# print(out)

	# net.zero_grad() # Clears the gradients as backward function will accumlate the gradients
	# out.backward(torch.randn(1,10))

	# Forward prop
	y = net(input)
	# Randomly initialize target values
	target = torch.randn(10)
	# Make it into a 1 by 10 vector
	target = target.view(1, -1)
	# Define Mean Squared Error Loss
	criterion = nn.MSELoss()
	# Compute the loss based on the output
	loss = criterion(y, target)

	# To see what was the previous method being called
	print(loss.grad_fn) # MSELoss
	print(loss.grad_fn.next_functions[0][0]) # Linear
	print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLu

	# Clear gradient buffers
	net.zero_grad()

	print('conv1.bias.grad before backward')
	print(net.conv1.bias.grad)

	loss.backward()

	print('conv1.bias.grad after backward')
	print(net.conv1.bias.grad)


	# Simple implementation of Stochastic Gradient Descent
	learning_rate = 0.01
	for f in net.parameters():
		f.data.sub_(f.grad.data * learning_rate) # Updating of weights