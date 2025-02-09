import torch
from torch import nn

# Define the neural network architecture
class SimpleNN(nn.Module):
	def __init__(self, in_size=1, out_size=1, hidden_size=5, hidden_layers=2, activation=nn.LeakyReLU):
		super().__init__()
		self.layers = nn.Sequential(
			nn.Linear(in_size, hidden_size),
			activation(),
			*[nn.Sequential(nn.Linear(hidden_size, hidden_size), activation()) for _ in range(hidden_layers)],
			nn.Linear(hidden_size, out_size)
		)


	def forward(self, x):
		return self.layers(x)
	

# recursive neural network that iterates over the same set of layers multiple times
class RecurrentNN(nn.Module):
	def __init__(self, in_size=1, out_size=1, hidden_size=5, hidden_layers=2, iterations=5, activation=nn.LeakyReLU):
		super().__init__()
		self.iterations = iterations
		self.layers = nn.Sequential(
			nn.Linear(in_size, hidden_size),
			activation(),
			*[nn.Sequential(nn.Linear(hidden_size, hidden_size), activation()) for _ in range(hidden_layers)],
			nn.Linear(hidden_size, out_size),
			# nn.Tanh(),
		)

	def forward(self, x):
		for _ in range(self.iterations):
			x = self.layers(x)
		return x
    

class SkipConn(nn.Module):
	def __init__(self, in_size=1, out_size=1, hidden_size=100, hidden_layers=7, activation=nn.LeakyReLU):
		super(SkipConn,self).__init__()
		end_size = hidden_size

		self.inLayer = nn.Linear(in_size, end_size)
		self.activate = activation()
		hidden = []
		for i in range(hidden_layers):
			start_size = end_size*2 + in_size if i>0 else end_size + in_size
			hidden.append(nn.Linear(start_size, end_size))
		self.hidden = nn.ModuleList(hidden)
		self.outLayer = nn.Linear(end_size*2+in_size, out_size)

	def forward(self, x):
		cur = self.activate(self.inLayer(x))
		prev = torch.tensor([])
		for layer in self.hidden:
			combined = torch.cat([cur, prev, x], 1)
			prev = cur
			cur = self.activate(layer(combined))
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


class AdaptiveExpertNN(nn.Module):
	def __init__(self, in_size=1, out_size=1, hidden_size=64, num_experts=4, fourier_order=4):
		super(AdaptiveExpertNN, self).__init__()
		self.fourier_order = fourier_order
		self.num_experts = num_experts
		
		# Feature processing
		self.feature_dim = in_size * (2 * fourier_order + 1)  # Fourier features + original
		self.orders = nn.Parameter(torch.arange(1, fourier_order + 1).float())
		
		# Gating network
		self.gate = nn.Sequential(
			nn.Linear(self.feature_dim, hidden_size),
			nn.LayerNorm(hidden_size),
			nn.LeakyReLU(),
			nn.Linear(hidden_size, num_experts),
			nn.Softmax(dim=-1)
		)
		
		# Expert networks with residual blocks
		self.experts = nn.ModuleList([
			nn.Sequential(
				ResidualBlock(self.feature_dim, hidden_size),
				ResidualBlock(hidden_size, hidden_size),
				ResidualBlock(hidden_size, hidden_size),
				nn.Linear(hidden_size, out_size)
			) for _ in range(num_experts)
		])
		
		# Adaptive scaling for each expert
		self.expert_scales = nn.Parameter(torch.ones(num_experts))
		
	def forward(self, x):
		# Generate enhanced feature space
		x = x.unsqueeze(-1)
		fourier_features = torch.cat([
			torch.sin(self.orders * x),
			torch.cos(self.orders * x),
			x
		], dim=-1)
		features = fourier_features.view(x.shape[0], -1)
		
		# Calculate expert weights
		expert_weights = self.gate(features)
		
		# Compute weighted sum of expert outputs
		output = torch.zeros(x.shape[0], 1, device=x.device)
		for i in range(self.num_experts):
			expert_out = self.experts[i](features)
			output += expert_weights[:, i:i+1] * expert_out * self.expert_scales[i]
		
		return output


class ResidualBlock(nn.Module):
	def __init__(self, in_size, out_size):
		super(ResidualBlock, self).__init__()
		self.linear1 = nn.Linear(in_size, out_size)
		self.linear2 = nn.Linear(out_size, out_size)
		self.norm1 = nn.LayerNorm(out_size)
		self.norm2 = nn.LayerNorm(out_size)
		self.activation = nn.GELU()
		
		# Projection layer if dimensions don't match
		self.project = nn.Linear(in_size, out_size) if in_size != out_size else nn.Identity()
		
	def forward(self, x):
		identity = self.project(x)
		
		out = self.linear1(x)
		out = self.norm1(out)
		out = self.activation(out)
		
		out = self.linear2(out)
		out = self.norm2(out)
		
		return self.activation(out + identity)

