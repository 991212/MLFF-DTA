import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from collections import OrderedDict
from torch.nn.modules.batchnorm import _BatchNorm
import torch_geometric.nn as gnn
from einops.layers.torch import Reduce

'''drug graph feature processing module (MGCN)'''
class NodeLevelBatchNorm(_BatchNorm):
	r"""
	Applies Batch Normalization over a batch of graph data.csv.
	Shape:
		- Input: [batch_nodes_dim, node_feature_dim]
		- Output: [batch_nodes_dim, node_feature_dim]
	batch_nodes_dim: all nodes of a batch graph
	"""

	def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
				 track_running_stats=True):
		super(NodeLevelBatchNorm, self).__init__(
			num_features, eps, momentum, affine, track_running_stats)

	def _check_input_dim(self, input):
		if input.dim() != 2:
			raise ValueError('expected 2D input (got {}D input)'
							 .format(input.dim()))

	def forward(self, input):
		self._check_input_dim(input)
		if self.momentum is None:
			exponential_average_factor = 0.0
		else:
			exponential_average_factor = self.momentum
		if self.training and self.track_running_stats:
			if self.num_batches_tracked is not None:
				self.num_batches_tracked = self.num_batches_tracked + 1
				if self.momentum is None:
					exponential_average_factor = 1.0 / float(self.num_batches_tracked)
				else:
					exponential_average_factor = self.momentum

		return torch.functional.F.batch_norm(
			input, self.running_mean, self.running_var, self.weight, self.bias,
			self.training or not self.track_running_stats,
			exponential_average_factor, self.eps)

	def extra_repr(self):
		return 'num_features={num_features}, eps={eps}, ' \
			   'affine={affine}'.format(**self.__dict__)

class GraphConvBn(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.conv = gnn.GraphConv(in_channels, out_channels)
		self.norm = NodeLevelBatchNorm(out_channels)

	def forward(self, data):
		# torch.Size([18888, 456]) torch.Size([2, 40866]) torch.Size([18888])
		x, edge_index, batch = data.x, data.edge_index, data.batch
		data.x = F.relu(self.norm(self.conv(x, edge_index)))

		return data

class DenseLayer(nn.Module):
	def __init__(self, num_input_features, growth_rate=32, bn_size=4):
		super().__init__()
		self.conv1 = GraphConvBn(num_input_features, int(growth_rate * bn_size))
		self.conv2 = GraphConvBn(int(growth_rate * bn_size), growth_rate)

	def bn_function(self, data):
		concated_features = torch.cat(data.x, 1)
		data.x = concated_features

		data = self.conv1(data)

		return data

	def forward(self, data):
		if isinstance(data.x, Tensor):
			data.x = [data.x]

		data = self.bn_function(data)
		data = self.conv2(data)

		return data


class DenseBlock(nn.ModuleDict):
	def __init__(self, num_layers, num_input_features, growth_rate=32, bn_size=4):
		super().__init__()
		for i in range(num_layers):
			layer = DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size)
			self.add_module('layer%d' % (i + 1), layer)

	def forward(self, data):
		features = [data.x]
		for name, layer in self.items():
			data = layer(data)
			features.append(data.x)
			data.x = features

		data.x = torch.cat(data.x, 1)

		return data


class Drug_Graph_Representation(nn.Module):
	def __init__(self, num_input_features, out_dim, growth_rate=32, block_config=(3, 3, 3, 3), bn_sizes=[2, 3, 4, 4]):
		super().__init__()
		self.features = nn.Sequential(OrderedDict([('conv0', GraphConvBn(num_input_features, 32))]))
		num_input_features = 32

		for i, num_layers in enumerate(block_config):
			block = DenseBlock(
				num_layers, num_input_features, growth_rate=growth_rate, bn_size=bn_sizes[i]
			)
			self.features.add_module('block%d' % (i + 1), block)
			num_input_features += int(num_layers * growth_rate)

			trans = GraphConvBn(num_input_features, num_input_features // 2)
			self.features.add_module("transition%d" % (i + 1), trans)
			num_input_features = num_input_features // 2

		self.classifer = nn.Linear(num_input_features, out_dim)

	def forward(self, data):
		data = self.features(data)
		x = gnn.global_mean_pool(data.x, data.batch)
		x = self.classifer(x)

		return x


'''drug semantics feature processing module (RDCN)'''
class ResDilaCNNBlock(nn.Module):
	def __init__(self, dilaSize, filterSize=256, dropout=0.15, name='ResDilaCNNBlock'):
		super(ResDilaCNNBlock, self).__init__()
		self.layers = nn.Sequential(
			nn.ReLU(),
			nn.Conv1d(filterSize, filterSize, kernel_size=3, padding=dilaSize, dilation=dilaSize),
			nn.ReLU(),
			nn.Conv1d(filterSize, filterSize, kernel_size=3, padding=dilaSize, dilation=dilaSize),
		)
		self.name = name

	def forward(self, x):
		# x: batchSize × filterSize × seqLen
		return x + self.layers(x)

class Drug_Semantics_Representation(nn.Module):
	# def __init__(self, feaSize, filterSize, blockNum=5, dropout=0.35, name='ResDilaCNNBlocks'):
	def __init__(self, feaSize, filterSize, blockNum=5, dilaSizeList=[1, 2, 4, 8, 16], dropout=0.5,
				 name='ResDilaCNNBlocks'):
		super(Drug_Semantics_Representation, self).__init__()  #
		self.blockLayers = nn.Sequential()
		self.linear = nn.Linear(feaSize, filterSize)
		for i in range(blockNum):
			self.blockLayers.add_module(f"ResDilaCNNBlock{i}",
										ResDilaCNNBlock(dilaSizeList[i % len(dilaSizeList)], filterSize,
														dropout=dropout))
		# self.blockLayers.add_module(f"ResDilaCNNBlock{i}", ResDilaCNNBlock(filterSize,dropout=dropout))
		self.name = name
		self.act = nn.ReLU()

	def forward(self, x):
		# x: batchSize × seqLen × feaSize
		x = self.linear(x)  # => batchSize × seqLen × filterSize
		x = self.blockLayers(x.transpose(1, 2))  # => batchSize × seqLen × filterSize
		x = self.act(x)  # => batchSize × seqLen × filterSize

		# x = self.pool(x.transpose(1, 2))
		x = Reduce('b c t -> b c', 'max')(x)
		return x


'''drug morgen fingerprint feature processing module(FCNN)'''
class Drug_MorganFP_Representation(nn.Sequential):
	def __init__(self, input_dim, output_dim, hidden_dims_lst):
		'''
			input_dim (int)
			output_dim (int)
			hidden_dims_lst (list, each element is a integer, indicating the hidden size)

		'''
		super(Drug_MorganFP_Representation, self).__init__()
		layer_size = len(hidden_dims_lst) + 1
		dims = [input_dim] + hidden_dims_lst + [output_dim]

		self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(layer_size)])

	def forward(self, v):
		v = v.float()
		for i, l in enumerate(self.predictor):
			v = F.relu(l(v))
		return v


'''drug mixed fingerprint feature processing module (fully connected layer) (FP-GNN)'''
class Drug_MixedFP_Representation(nn.Module):
	def __init__(self,hp):
		super(Drug_MixedFP_Representation, self).__init__()
		self.fp_2_dim = hp.mixed_fp_2_dim # The dim of the second layer in fpn
		self.dropout_fpn = hp.mixed_fp_dropput
		self.hidden_dim = hp.module_out_dim

		fp_dim = 1489
		self.fc1 = nn.Linear(fp_dim, self.fp_2_dim)
		self.act_func = nn.ReLU()
		self.fc2 = nn.Linear(self.fp_2_dim, self.hidden_dim)
		self.dropout = nn.Dropout(p=self.dropout_fpn)


	def forward(self,mixed_fp):
		mixed_fp = mixed_fp.to(self.fc1.weight.dtype)
		fpn_out = self.fc1(mixed_fp)
		fpn_out = self.dropout(fpn_out)
		fpn_out = self.act_func(fpn_out)
		fpn_out = self.fc2(fpn_out)
		return fpn_out



