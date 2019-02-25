"""
Credit to jaesik817 for the physics engine.
Original implementation:
https://github.com/jaesik817/Interaction-networks_tensorflow/blob/master/physics_engine.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from math import cos, pi, radians, sin

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

# number of time steps
ts = 1000

# number of features, [mass, x, y, vx, vy]
num_features = 5

# G
G = 10 ** 5

# length of each time interval
dt = 0.001


def init(n_body, fea_num, orbit):
	"""
	Initialization on just the first time step; fill all other time steps with 0's

	:param n_body: number of objects
	:param fea_num: number of features
	:param orbit: whether simulate planet orbit
	:return: a numpy vector of shape (ts, n_body, fea_num)
	"""
	data = np.zeros((ts, n_body, fea_num), dtype=float)
	if orbit:
		data[0][0][0] = 100
		data[0][0][1:5] = 0.0
		for i in range(1, n_body):
			data[0][i][0] = np.random.rand() * 8.98 + 0.02
			distance = np.random.rand() * 90.0 + 10.0
			theta = np.random.rand() * 360
			theta_rad = pi / 2 - radians(theta)
			data[0][i][1] = distance * cos(theta_rad)
			data[0][i][2] = distance * sin(theta_rad)
			data[0][i][3] = -1 * data[0][i][2] / norm(data[0][i][1:3]) * (
					G * data[0][0][0] / norm(data[0][i][1:3]) ** 2) * distance / 1000
			data[0][i][4] = data[0][i][1] / norm(data[0][i][1:3]) * (
					G * data[0][0][0] / norm(data[0][i][1:3]) ** 2) * distance / 1000
	# data[0][i][3]=np.random.rand()*10.0-5.0
	# data[0][i][4]=np.random.rand()*10.0-5.0
	else:
		for i in range(n_body):
			data[0][i][0] = np.random.rand() * 8.98 + 0.02  # mass
			distance = np.random.rand() * 90.0 + 10.0
			theta = np.random.rand() * 360
			theta_rad = pi / 2 - radians(theta)
			data[0][i][1] = distance * cos(theta_rad)  # x pos
			data[0][i][2] = distance * sin(theta_rad)  # y pos
			data[0][i][3] = np.random.rand() * 6.0 - 3.0  # x vel
			data[0][i][4] = np.random.rand() * 6.0 - 3.0  # y vel
	return data


def norm(x):
	return np.sqrt(np.sum(x ** 2))


def get_f(receiver, sender):
	"""
	Return gravitational force between two bodies (in vector form).
	F = G*m1*m2 / r**2
	"""
	diff = sender[1:3] - receiver[1:3]  # difference in (x, y)
	distance = norm(diff)
	if distance < 1:
		distance = 1
	return G * receiver[0] * sender[0] / (distance ** 3) * diff


def calc(cur_state, n_body):
	"""
	Given current states of n objects, calculate their next states.

	:return: a numpy vector of shape (n_body, num_features)
	"""
	next_state = np.zeros((n_body, num_features), dtype=float)
	f_mat = np.zeros((n_body, n_body, 2), dtype=float)
	f_sum = np.zeros((n_body, 2), dtype=float)
	acc = np.zeros((n_body, 2), dtype=float)
	for i in range(n_body):
		for j in range(i + 1, n_body):
			if j != i:
				# i is receiver, j is sender
				f = get_f(cur_state[i][:3], cur_state[j][:3])
				f_mat[i, j] += f
				f_mat[j, i] -= f
		f_sum[i] = np.sum(f_mat[i], axis=0)
		acc[i] = f_sum[i] / cur_state[i][0]  # F = ma
		next_state[i][0] = cur_state[i][0]
		next_state[i][3:5] = cur_state[i][3:5] + acc[i] * dt
		next_state[i][1:3] = cur_state[i][1:3] + next_state[i][3:5] * dt
	return next_state


def gen(n_body, orbit):
	# initialize the first time step
	d = init(n_body, num_features, orbit)

	# calculate data for remaining time steps
	for i in range(1, ts):
		d[i] = calc(d[i - 1], n_body)
	return d


def make_video(xy, filename):
	os.system("rm -rf pics/*")
	FFMpegWriter = manimation.writers['ffmpeg']
	metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
	writer = FFMpegWriter(fps=15, metadata=metadata)
	fig = plt.figure()
	plt.xlim(-200, 200)
	plt.ylim(-200, 200)
	fig_num = len(xy)
	color = ['ro', 'bo', 'go', 'ko', 'yo', 'mo', 'co']
	with writer.saving(fig, filename, fig_num):
		for i in range(fig_num):
			for j in range(len(xy[0])):
				plt.plot(xy[i, j, 1], xy[i, j, 0], color[j % len(color)])
			writer.grab_frame()


if __name__ == '__main__':
	data = gen(3, True)
	xy = data[:, :, 1:3]  # x, y positions
	make_video(xy, "test.mp4")
