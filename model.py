import torch
import torch.nn as nn

# constants

n_objects = 3
obj_dim = 7  # mass, x pos, y pos, x speed, y speed

n_relations = n_objects * (n_objects - 1)
rel_dim = 1

eff_dim = 100
hidden_obj_dim = 100
hidden_rel_dim = 100


class RelationModel(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(RelationModel, self).__init__()

		self.model = nn.Sequential(
			nn.Linear(input_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, output_size),
			nn.ReLU()
		)

	def forward(self, x):
		'''
        Args:
            x: [n_relations, input_size]
        Returns:
            [n_relations, output_size]
        '''
		return self.model(x)


class ObjectModel(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(ObjectModel, self).__init__()

		self.model = nn.Sequential(
			nn.Linear(input_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, output_size)
		)

	def forward(self, x):
		'''
        Args:
            x: [n_objects, input_size]
        Returns:
            [n_objects, output_size]

        Note: output_size = number of states we want to predict
        '''
		return self.model(x)


class InteractionNetwork(nn.Module):
	def __init__(self, n_obj, n_rel, dim_obj, dim_rel, dim_eff, dim_hidden_obj, dim_hidden_rel):
		super(InteractionNetwork, self).__init__()
		self.rm = RelationModel(dim_obj * 2 + dim_rel, dim_hidden_rel, dim_eff)
		self.om = ObjectModel(dim_obj + dim_eff, dim_hidden_obj, 3)  # x, y, z

	def m(self, obj, rr, rs, ra):
		orr = obj.mm(rr)
		ors = obj.mm(rs)
		return torch.cat([orr, ors, ra])

	def forward(self, obj, rr, rs, ra, x):
		# marshalling function
		b = m(obj, rr, rs, ra)

		# relation module
		e = self.rm(b)

		# effect aggregator
		a = torch.cat([obj, x, e.mm(rr.t())])

		# object module
		p = self.om(c)

		return p


interaction_network = \
	InteractionNetwork(n_objects, n_relations, obj_dim, rel_dim, eff_dim, hidden_obj_dim, hidden_rel_dim)
