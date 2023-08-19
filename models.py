import torch
from torch import nn

# Define the neural network architecture
class SimpleNN(nn.Module):
	def __init__(self, in_size=1, out_size=1, hidden_size=5, hidden_layers=2):
		super().__init__()
		self.layers = nn.Sequential(
			nn.Linear(in_size, hidden_size),
			nn.ReLU(),
			*[nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.LeakyReLU()) for _ in range(hidden_layers)],
			nn.Linear(hidden_size, out_size)
		)

	def forward(self, x):
		return self.layers(x)
    

class SkipConn(nn.Module):
	def __init__(self, in_size=1, out_size=1, hidden_size=100, hidden_layers=7):
		super(SkipConn,self).__init__()
		end_size = hidden_size

		self.inLayer = nn.Linear(in_size, end_size)
		self.relu = nn.LeakyReLU()
		hidden = []
		for i in range(hidden_layers):
			start_size = end_size*2 + in_size if i>0 else end_size + in_size
			hidden.append(nn.Linear(start_size, end_size))
		self.hidden = nn.ModuleList(hidden)
		self.outLayer = nn.Linear(end_size*2+in_size, out_size)

	def forward(self, x):
		cur = self.relu(self.inLayer(x))
		prev = torch.tensor([])
		for layer in self.hidden:
			combined = torch.cat([cur, prev, x], 1)
			prev = cur
			cur = self.relu(layer(combined))
		return self.outLayer(torch.cat([cur, prev, x], 1))
		# return self.sig(y)


class Fourier(nn.Module):
	def __init__(self, in_size=1, out_size=1, fourier_order=4, hidden_size=100, hidden_layers=7):
		super(Fourier,self).__init__()
		self.fourier_order = fourier_order
		in_size = in_size * 2 * fourier_order + in_size
		self.inner_model = SkipConn(in_size=in_size, out_size=out_size, hidden_size=hidden_size, hidden_layers=hidden_layers)
		self.orders = torch.arange(1, fourier_order + 1).float()

	def forward(self,x):
		x = x.unsqueeze(-1)  # add an extra dimension for broadcasting
		fourier_features = torch.cat([torch.sin(self.orders * x), torch.cos(self.orders * x), x], dim=-1)
		fourier_features = fourier_features.view(x.shape[0], -1)  # flatten the last two dimensions
		return self.inner_model(fourier_features)
	

class SimpleTaylorNN(nn.Module):
	# computes inputs to the network as terms of the taylor series up to a given order, then passes them through a sinlge linear layer without a bias or activation function
	def __init__(self, in_size=1, out_size=1, taylor_order=4):
		super(SimpleTaylorNN,self).__init__()
		self.taylor_order = taylor_order
		self.linear = nn.Linear(taylor_order, out_size, bias=False)
		self.orders = torch.arange(0, taylor_order).float()

	def forward(self,x):
		x = x.unsqueeze(-1)
		taylor_features = torch.pow(x, self.orders)
		taylor_features = taylor_features.view(x.shape[0], -1)
		return self.linear(taylor_features)


class TaylorNN(nn.Module):
	def __init__(self, in_size=1, out_size=1, taylor_order=4, hidden_size=100, hidden_layers=7):
		super(TaylorNN,self).__init__()
		self.taylor_order = taylor_order
		in_size = in_size * taylor_order
		self.inner_model = SimpleNN(in_size=in_size, out_size=out_size, hidden_size=hidden_size, hidden_layers=hidden_layers)
		self.orders = torch.arange(1, taylor_order + 1).float()

	def forward(self,x):
		x = x.unsqueeze(-1)
		taylor_features = torch.pow(x, self.orders)
		taylor_features = taylor_features.view(x.shape[0], -1)
		return self.inner_model(taylor_features)