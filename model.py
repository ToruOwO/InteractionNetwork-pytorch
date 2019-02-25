import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from physics import gen

# constants

n_objects = 3
obj_dim = 5  # mass, x pos, y pos, x speed, y speed

n_relations = n_objects * (n_objects - 1)
rel_dim = 1

eff_dim = 100
hidden_obj_dim = 100
hidden_rel_dim = 100

data = gen(n_objects, True)


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
		self.om = ObjectModel(dim_obj + dim_eff, dim_hidden_obj, 2)  # x, y

	def m(self, obj, rr, rs, ra):
		orr = obj.mm(rr)
		ors = obj.mm(rs)
		return torch.cat([orr, ors, ra])

	def forward(self, obj, rr, rs, ra, x):
		# marshalling function
		b = self.m(obj, rr, rs, ra)

		# relation module
		e = self.rm(b)

		# effect aggregator
		a = torch.cat([obj, x, e.mm(rr.t())])

		# object module
		p = self.om(a)

		return p


def get_data(idx):
	objs = data[idx, :, :, :]

	receiver_r = np.zeros((n_objects, n_relations), dtype=float)
	sender_r = np.zeros((n_objects, n_relations), dtype=float)

	count = 0
	for i in range(n_objects):
		for j in range(n_objects):
			if i != j:
				receiver_r[i, count] = 1.0
				sender_r[j, count] = 1.0
				count += 1

	r_info = np.zeros((n_relations, rel_dim))
	target = data[idx+1, :, :, 3:]   # only want vx and vy predictions

	return objs, sender_r, receiver_r, r_info, target

# set up network
interaction_network = \
	InteractionNetwork(n_objects, n_relations, obj_dim, rel_dim, eff_dim, hidden_obj_dim, hidden_rel_dim)

optimizer = optim.Adam(interaction_network.parameters())
criterion = nn.MSELoss()

# training
n_epoch = 100

losses = []
for epoch in range(n_epoch):
	for i in range(len(data)):
		objects, sender_relations, receiver_relations, relation_info, target = get_data(i)
		predicted = interaction_network(objects, sender_relations, receiver_relations, relation_info)
		loss = criterion(predicted, target)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		losses.append(np.sqrt(loss.data[0]))

	import matplotlib.pyplot as plt
	plt.figure(figsize=(20, 5))
	plt.subplot(131)
	plt.title('Epoch %s RMS Error %s' % (epoch, np.sqrt(np.mean(losses[-100:]))))
	plt.plot(losses)
	plt.show()