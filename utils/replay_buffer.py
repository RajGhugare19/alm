import numpy as np

class ReplayMemory():
    def __init__(self, buffer_limit, obs_size, action_size, obs_dtype):
        print('buffer limit is = ', buffer_limit)
        self.buffer_limit = buffer_limit
        self.observation = np.empty((buffer_limit, obs_size), dtype=obs_dtype) 
        self.next_observation = np.empty((buffer_limit, obs_size), dtype=obs_dtype) 
        self.action = np.empty((buffer_limit, action_size), dtype=np.float32)
        self.reward = np.empty((buffer_limit,), dtype=np.float32) 
        self.terminal = np.empty((buffer_limit,), dtype=bool)
        self.idx = 0
        self.full = False

    def push(self, transition):
        state, action, reward, next_state, done = transition
        self.observation[self.idx] = state
        self.next_observation[self.idx] = next_state
        self.action[self.idx] = action 
        self.reward[self.idx] = reward
        self.terminal[self.idx] = done
        self.idx = (self.idx + 1) % self.buffer_limit
        self.full = self.full or self.idx == 0
    
    def sample(self, n):
        idxes = np.random.randint(0, self.buffer_limit if self.full else self.idx, size=n)
        return self.observation[idxes], self.action[idxes], self.reward[idxes], self.next_observation[idxes], self.terminal[idxes]

    def sample_seq(self, seq_len, batch_size):
        n = batch_size
        l = seq_len
        obs, act, rew, next_obs, term = self._retrieve_batch(np.asarray([self._sample_idx(l) for _ in range(n)]), n, l)
        return obs, act, rew, next_obs, term

    def sample_probe_data(self, data_size):
        idxes = np.random.randint(0, self.buffer_limit if self.full else self.idx, size=data_size)
        return self.observation[idxes]

    def _sample_idx(self, L):
        valid_idx = False 
        while not valid_idx:
            idx = np.random.randint(0, self.buffer_limit if self.full else self.idx-L)
            idxs = np.arange(idx, idx+L)%self.buffer_limit
            valid_idx = (not self.idx in idxs[1:]) and (not self.terminal[idxs[:-1]].any())
        return idxs 

    def _retrieve_batch(self, idxs, n, l):
        vec_idxs = idxs.transpose().reshape(-1)
        return self.observation[vec_idxs].reshape(l, n, -1), self.action[vec_idxs].reshape(l, n, -1), self.reward[vec_idxs].reshape(l, n), \
            self.next_observation[vec_idxs].reshape(l, n, -1), self.terminal[vec_idxs].reshape(l, n)
    
    def __len__(self):
        return self.buffer_limit if self.full else self.idx+1