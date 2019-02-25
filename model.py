import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from physics import gen

# constants

n_objects = 3
obj_dim = 5  # mass, x pos, y pos, x speed, y speed

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
	def __init__(self, dim_obj, dim_rel, dim_eff, dim_hidden_obj, dim_hidden_rel, dim_x=0):
		super(InteractionNetwork, self).__init__()
		self.rm = RelationModel(dim_obj * 2 + dim_rel, dim_hidden_rel, dim_eff)
		self.om = ObjectModel(dim_obj + dim_eff + dim_x, dim_hidden_obj, 2)  # x, y

	def m(self, obj, rr, rs, ra):
		"""
		The marshalling function;
		computes the matrix products ORr and ORs and concatenates them with Ra

		:param obj: object states
		:param rr: receiver relations
		:param rs: sender relations
		:param ra: relation info
		:return:
		"""
		orr = obj.t().mm(rr)   # (obj_dim, n_relations)
		ors = obj.t().mm(rs)   # (obj_dim, n_relations)
		return torch.cat([orr, ors, ra.t()])   # (obj_dim*2+rel_dim, n_relations)

	def forward(self, obj, rr, rs, ra, x=None):
		"""
		objects, sender_relations, receiver_relations, relation_info
		:param obj: (n_objects, obj_dim)
		:param rr: (n_objects, n_relations)
		:param rs: (n_objects, n_relations)
		:param ra: (n_relations, rel_dim)
		:param x: external forces, default to None
		:return:
		"""
		# marshalling function
		b = self.m(obj, rr, rs, ra)   # shape of b = (obj_dim*2+rel_dim, n_relations)

		# relation module
		e = self.rm(b.t())   # shape of e = (n_relations, eff_dim)
		e = e.t()   # shape of e = (eff_dim, n_relations)

		# effect aggregator
		if x is None:
			a = torch.cat([obj.t(), e.mm(rr.t())])   # shape of a = (obj_dim+eff_dim, n_objects)
		else:
			a = torch.cat([obj.t(), x, e.mm(rr.t())])   # shape of a = (obj_dim+ext_dim+eff_dim, n_objects)

		# object module
		p = self.om(a.t())   # shape of p = (n_objects, 2)

		return p


def format_data(data, idx):
	objs = data[idx, :, :]   # (n_objects, obj_dim)

	receiver_r = np.zeros((n_objects, n_relations), dtype=float)
	sender_r = np.zeros((n_objects, n_relations), dtype=float)

	count = 0   # used as idx of relations
	for i in range(n_objects):
		for j in range(n_objects):
			if i != j:
				receiver_r[i, count] = 1.0
				sender_r[j, count] = 1.0
				count += 1

	r_info = np.zeros((n_relations, rel_dim))
	target = data[idx + 1, :, 3:]  # only want vx and vy predictions

	objs = Variable(torch.FloatTensor(objs))
	sender_r = Variable(torch.FloatTensor(sender_r))
	receiver_r = Variable(torch.FloatTensor(receiver_r))
	r_info = Variable(torch.FloatTensor(r_info))
	target = Variable(torch.FloatTensor(target))

	return objs, sender_r, receiver_r, r_info, target


# set up network
interaction_network = \
	InteractionNetwork(obj_dim, rel_dim, eff_dim, hidden_obj_dim, hidden_rel_dim)

optimizer = optim.Adam(interaction_network.parameters())
criterion = nn.MSELoss()

# training
n_epoch = 100

losses = []

# generate orbiting planets data
data = gen(n_objects, True)   # shape of data = (ts, n_objects, object_dim)

for epoch in range(n_epoch):
	print("="*20, "epoch", epoch, "="*20)

	best_loss = np.inf
	for i in range(len(data)-1):
		objects, sender_relations, receiver_relations, relation_info, target = format_data(data, i)
		predicted = interaction_network(objects, sender_relations, receiver_relations, relation_info)
		loss = criterion(predicted, target)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		losses.append(np.sqrt(loss.data[0]))

		if losses[-1] < best_loss:
			best_loss = losses[-1]

	print("best loss:", best_loss)

	# plot losses for each time step
	import matplotlib.pyplot as plt

	plt.figure(figsize=(20, 5))
	plt.subplot(131)
	plt.title('Epoch %s RMS Error %s' % (epoch, np.sqrt(np.mean(losses[-100:]))))
	plt.plot(losses)
	plt.savefig('epoch_{}.png'.format(epoch))
