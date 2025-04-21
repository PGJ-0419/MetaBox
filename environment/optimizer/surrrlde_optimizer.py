import torch
import numpy as np
from collections import deque
from ..optimizer.learnable_optimizer import Learnable_Optimizer


class SurrRLDE_Optimizer(Learnable_Optimizer):
	def __init__(self, config):
		super().__init__(config)

		config.F = 0.5
		config.Cr = 0.7
		config.NP = 100
		self.device = config.device
		self.config = config

		self.F = config.F
		self.Cr = config.Cr
		self.pop_size = config.NP
		self.maxFEs = config.maxFEs
		self.dim = config.dim
		self.ub = config.upperbound
		self.lb = -config.upperbound

		self.population = None
		self.fitness = None
		self.pop_cur_best = None
		self.fit_cur_best = None
		self.pop_history_best = None
		self.fit_history_best = None
		self.fit_init_best = None

		self.improved_gen = 0

		self.fes = None  # record the number of function evaluations used
		self.cost = None
		self.cur_logpoint = None  # record the current logpoint
		self.log_interval = config.log_interval

	def __str__(self):
		return "SurrRLDE_Optimizer"

	def get_state(self, problem):
		state = torch.zeros(9)
		# state 1
		diff = self.population.unsqueeze(0) - self.population.unsqueeze(1)
		distances = torch.sqrt(torch.sum(diff ** 2, dim=2))
		state[0] = torch.sum(distances) / (self.population.shape[0] * (self.population.shape[0] - 1))

		# state 2
		diff = self.population - self.pop_cur_best
		distances = torch.sqrt(torch.sum(diff ** 2, dim=1))
		state[1] = torch.sum(distances) / (self.population.shape[0])

		# state 3
		diff = self.population - self.pop_history_best
		distances = torch.sqrt(torch.sum(diff ** 2, dim=1))
		state[2] = torch.sum(distances) / (self.population.shape[0])

		# state 4
		diff = self.fitness - self.fit_history_best
		distances = torch.sqrt(torch.sum(diff ** 2, dim=-1))
		state[3] = torch.sum(distances) / (self.fitness.shape[0])

		# state 5
		diff = self.fitness - self.fit_cur_best
		# print

		distances = torch.sqrt(torch.sum(diff ** 2, dim=-1))
		state[3] = torch.sum(distances) / (self.fitness.shape[0])

		# state 6 std(y)
		state[5] = torch.std(self.fitness)

		# state 7 (T - t)/T
		state[6] = (self.maxFEs - self.fes) / self.maxFEs

		# state 8
		if self.fit_cur_best < self.fit_history_best:
			self.improved_gen = 0
		else:
			self.improved_gen += 1

		state[7] = self.improved_gen

		# state 9 bool
		if self.fit_cur_best < self.fit_history_best:
			state[8] = 1
		else:
			state[8] = 0
		return state

	def init_population(self, problem):
		self.rng_torch = self.rng_cpu
		if self.device != "cpu":
			self.rng_torch = self.rng_gpu

		self.population = (torch.rand(self.pop_size, self.dim, generator = self.rng_torch, device = self.device)
						   * (problem.ub - problem.lb) + problem.lb)
		#(-5,5)
		self.population = self.population.to(self.device)

		if self.config.train_problem in ['bbob-surrogate-10D','bbob-surrogate-5D','bbob-surrogate-2D'] and self.config.is_train:
			# print(self.population.clone().to(self.device))
			self.fitness = problem.eval(self.population.clone().to(self.device))

		else:
			if problem.optimum is None:
				self.fitness = problem.eval(self.population.clone().cpu().numpy())
			else:
				self.fitness = problem.eval(self.population.clone().cpu().numpy()) - problem.optimum

		if isinstance(self.fitness, np.ndarray):
			self.fitness = torch.from_numpy(self.fitness).to(self.device)
		if self.fitness.shape == (self.pop_size,):
			self.fitness = self.fitness.unsqueeze(1)

		self.pop_cur_best = self.population[torch.argmin(self.fitness)].clone()
		self.pop_history_best = self.population[torch.argmin(self.fitness)].clone()


		self.fit_init_best = torch.min(self.fitness).clone()
		self.fit_cur_best = torch.min(self.fitness).clone()
		self.fit_history_best = torch.min(self.fitness).clone()

		self.fes = self.pop_size
		self.cost = [self.fit_cur_best.clone().cpu().item()]  # record the best cost of first generation
		self.cur_logpoint = 1  # record the current logpoint
		state = self.get_state(problem)
		if self.config.full_meta_data:
			self.meta_X = [self.population.clone().cpu().numpy()]
			self.meta_Cost = [self.fitness.clone().cpu().numpy()]

		return state

	def update(self, action, problem):
		'''
		F:0.1,0.5,0.9
		mutation: rand1,best1,current-rand,current-pbest(p = 10%),currend-best
		'''

		if action == 0:
			mut_way = 'DE/rand/1'
			self.F = 0.1
		elif action == 1:
			mut_way = 'DE/rand/1'
			self.F = 0.5
		elif action == 2:
			mut_way = 'DE/rand/1'
			self.F = 0.9
		elif action == 3:
			mut_way = 'DE/best/1'
			self.F = 0.1
		elif action == 4:
			mut_way = 'DE/best/1'
			self.F = 0.5
		elif action == 5:
			mut_way = 'DE/best/1'
			self.F = 0.9
		elif action == 6:
			mut_way = 'DE/current-to-rand'
			self.F = 0.1
		elif action == 7:
			mut_way = 'DE/current-to-rand'
			self.F = 0.5
		elif action == 8:
			mut_way = 'DE/current-to-rand'
			self.F = 0.9
		elif action == 9:
			mut_way = 'DE/current-to-pbest'
			self.F = 0.1
		elif action == 10:
			mut_way = 'DE/current-to-pbest'
			self.F = 0.5
		elif action == 11:
			mut_way = 'DE/current-to-pbest'
			self.F = 0.9
		elif action == 12:
			mut_way = 'DE/current-to-best'
			self.F = 0.1
		elif action == 13:
			mut_way = 'DE/current-to-best'
			self.F = 0.5
		elif action == 14:
			mut_way = 'DE/current-to-best'
			self.F = 0.9
		else:
			raise ValueError(f'action error: {action}')

		mut_population = self.mutation(mut_way)
		crossover_population = self.crossover(mut_population)

		if self.config.train_problem in ['bbob-surrogate-10D', 'bbob-surrogate-5D','bbob-surrogate-2D'] and self.config.is_train:
			temp_fit = problem.eval(crossover_population.clone().to(self.device))
		else:
			if problem.optimum is None:
				temp_fit = problem.eval(crossover_population.clone().cpu().numpy())
			else:
				temp_fit = problem.eval(crossover_population.clone().cpu().numpy()) - problem.optimum

		if isinstance(temp_fit, np.ndarray):
			temp_fit = torch.from_numpy(temp_fit).to(self.device)
		if temp_fit.shape == (self.pop_size,):
			temp_fit = temp_fit.unsqueeze(1)

		for i in range(self.pop_size):
			if temp_fit[i].item() < self.fitness[i].item():
				self.fitness[i] = temp_fit[i]
				self.population[i] = crossover_population[i]

		reward = self.fit_history_best > torch.min(self.fit_history_best, torch.min(self.fitness).clone())

		best_index = torch.argmin(self.fitness)

		self.pop_cur_best = self.population[best_index].clone()
		self.fit_cur_best = self.fitness[best_index].clone()

		next_state = self.get_state(problem)

		if self.fit_cur_best < self.fit_history_best:
			self.fit_history_best = self.fit_cur_best.clone()
			self.pop_history_best = self.pop_cur_best.clone()

		is_done = (self.fes >= self.maxFEs)

		self.fes += self.pop_size

		if self.fes >= self.cur_logpoint * self.config.log_interval:
			self.cur_logpoint += 1
			self.cost.append(self.fit_history_best.clone().cpu().item())

		if self.config.full_meta_data:
			self.meta_X.append(self.population.clone().cpu().numpy())
			self.meta_Cost.append(self.fitness.clone().cpu().numpy())

		if is_done:
			if len(self.cost) >= self.config.n_logpoint + 1:
				self.cost[-1] = self.fit_history_best.clone().cpu().item()
			else:
				while len(self.cost) < self.__config.n_logpoint + 1:
					self.cost.append(self.fit_history_best.clone().cpu().item())

		info = {}
		return next_state, reward.item(), is_done, info

	def mutation(self, mut_way):
		mut_population = torch.zeros_like(self.population, device=self.device)

		if mut_way == 'DE/rand/1':

			r = self.generate_random_int(self.pop_size, 3)  # Shape: [pop_size, 3]
			a = self.population[r[:, 0]]
			b = self.population[r[:, 1]]
			c = self.population[r[:, 2]]

			v = a + self.F * (b - c)

			v = torch.clamp(v, min=self.lb, max=self.ub)
			mut_population = v

		elif mut_way == 'DE/best/1':
			r = self.generate_random_int(self.pop_size, 2)  # Shape: [pop_size, 2]
			a = self.population[r[:, 0]]
			b = self.population[r[:, 1]]
			v = self.pop_cur_best + self.F * (a - b)
			v = torch.clamp(v, self.lb, self.ub)
			mut_population = v

		elif mut_way == 'DE/current-to-rand':
			r = self.generate_random_int(self.pop_size, 3)  # Shape: [pop_size, 3]
			a = self.population[r[:, 0]]
			b = self.population[r[:, 1]]
			c = self.population[r[:, 2]]
			v = self.population + self.F * (a - self.population) + self.F * (b - c)
			v = torch.clamp(v, self.lb, self.ub)
			mut_population = v

		elif mut_way == 'DE/current-to-pbest':
			p = 0.1
			p_num = max(1, int(p * self.pop_size))
			sorted_indices = torch.argsort(self.fitness.clone().flatten())
			pbest_indices = sorted_indices[:p_num]
			r = self.generate_random_int(self.pop_size, 2)  # Shape: [pop_size, 2]

			a = self.population[r[:, 0]]
			b = self.population[r[:, 1]]

			pbest_index = pbest_indices[torch.randint(0, p_num, (self.pop_size,), generator = self.rng_torch, device = self.device)]
			pbest = self.population[pbest_index]

			v = self.population + self.F * (pbest - self.population) + self.F * (a - b)
			v = torch.clamp(v, self.lb, self.ub)
			mut_population = v

		elif mut_way == 'DE/current-to-best':
			r = self.generate_random_int(self.pop_size, 4)  # Shape: [pop_size, 4]
			a = self.population[r[:, 0]]
			b = self.population[r[:, 1]]
			c = self.population[r[:, 2]]
			d = self.population[r[:, 3]]
			v = self.population + self.F * (self.pop_cur_best - self.population) + self.F * (a - b) + self.F * (c - d)
			v = torch.clamp(v, self.lb, self.ub)
			mut_population = v

		else:
			raise ValueError(f'mutation error: {mut_way} is not defined')

		mut_population = torch.tensor(mut_population, device=self.device)
		return mut_population

	def crossover(self, mut_population):
		crossover_population = self.population.clone()
		for i in range(self.pop_size):

			select_dim = torch.randint(0, self.dim, (1,), generator = self.rng_torch, device = self.device)
			for j in range(self.dim):
				if torch.rand(1, generator = self.rng_torch, device = self.device) < self.Cr or j == select_dim:
					crossover_population[i][j] = mut_population[i][j]
		return crossover_population

	def generate_random_int(self, NP: int, cols: int) -> torch.Tensor:
		r = torch.randint(0, NP, (NP, cols), dtype = torch.long, generator = self.rng_torch, device = self.device)  # [NP, 3]

		for i in range(NP):
			while r[i, :].eq(i).any():
				r[i, :] = torch.randint(0, NP, (cols,), dtype = torch.long, generator = self.rng_torch, device = self.device)

		return r
