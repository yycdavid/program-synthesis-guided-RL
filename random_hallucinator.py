import numpy as np
class RandomHallucinator:
	def __init__(self, env_name, env):
		self.env_name = env_name
		self.env = env
		print(self.env_name)

	def decode(self, z):
		z = z.squeeze(0)
		cond_inp = z[100:]

		if self.env_name != "BoxWorld":
			unobs_fraction = cond_inp[-1]
			abstract_states = cond_inp[:-1-len(self.env.grabbable_indices)]

			abstract_state = abstract_states[:len(abstract_states)//2]

			new_abstract_state = abstract_state.clone()

			# figure out number of random items to add
			max_additions = int(10*unobs_fraction)
			num_additions = np.random.randint(0, max_additions + 1)

			# First three dimensions are region_exists_vec, boundary_vec
			things_vec = new_abstract_state[2 + 1:].reshape(-1, self.env.ABS_MAX_VAL)

			for i in range(num_additions):
				idx = np.random.randint(0, things_vec.size(0))
				# add one to the thing at this idx
				for k in range(things_vec.size(1)):
					if things_vec[idx][k] == 0:
						things_vec[idx][k] = 1
						break

			#print(self.env.parse_abstract_state(new_abstract_state))
			#print(self.env.parse_abstract_state(abstract_state))

			return new_abstract_state.unsqueeze(0)
