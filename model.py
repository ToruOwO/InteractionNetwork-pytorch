import torch
import torch.nn as nn

class RelationModel(nn.Module):
	def __init__(self):
		super(RelationModel, self).__init__()
		pass

	def forward(self, x):
		pass


class ObjectModel(nn.Module):
	"""docstring for ObjectModel"""
	def __init__(self, arg):
		super(ObjectModel, self).__init__()
		pass

	def forward(self, x):
		pass


class InteractionNetwork(nn.Module):
	def __init__(self, n_obj, n_rel, dim_obj, dim_rel, dim_eff):
		super(InteractionNetwork, self).__init__()
		self.rm = RelationModel()
		self.om = ObjectModel()

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
		a = torch.cat([obj, x, e.mm(rr.T)])

		# object module
		p = self.om(c)

		return p