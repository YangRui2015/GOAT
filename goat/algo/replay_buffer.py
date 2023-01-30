import threading
import pickle
import numpy as np
from goat.common import logger

class ReplayBuffer:
    def __init__(self, buffer_shapes, size_in_transitions, T, sample_transitions, default_sampler, info=None): 
        """Creates a replay buffer.
        Args:
            buffer_shapes (dict of ints): the shape for all buffers that are used in the replay
                buffer
            size_in_transitions (int): the size of the buffer, measured in transitions
            T (int): the time horizon for episodes
            sample_transitions (function): a function that samples from the replay buffer
        """
        self.buffer_shapes = buffer_shapes
        self.size = size_in_transitions // T
        self.training_size = self.size
        self.T = T
        self.sample_transitions = sample_transitions
        self.default_sampler = default_sampler
        self.info = info
        self.kde = self.info['kde'] if 'kde' in self.info.keys() else None
        # self.buffers is {key: array(size_in_episodes x T or T+1 x dim_key)}
        self.buffers = {key: np.empty([self.size, *shape])
                        for key, shape in buffer_shapes.items()}

        # memory management
        self.point = 0
        self.current_size = 0
        self.n_transitions_stored = 0
        self.lock = threading.Lock()
        
    @property
    def full(self):
        with self.lock:
            return self.current_size == self.size

    def sample(self, batch_size, random=False):
        """Returns a dict {key: array(batch_size x shapes[key])}
        """
        buffers = {}
        with self.lock:
            assert self.current_size > 0
            for key in self.buffers.keys():
                buffers[key] = self.buffers[key][:self.current_size]

        if 'o_2' not in buffers and 'ag_2' not in buffers:
            buffers['o_2'] = buffers['o'][:, 1:, :]
            buffers['ag_2'] = buffers['ag'][:, 1:, :]

        if random:
            transitions = self.default_sampler(buffers, batch_size, self.info)
        else:
            transitions = self.sample_transitions(buffers, batch_size, self.info)

        for key in (['r', 'o_2', 'ag_2'] + list(self.buffers.keys())):
            assert key in transitions, "key %s missing from transitions" % key

        return transitions

    def store_episode(self, episode_batch):
        """episode_batch: array(rollout_batch_size x (T or T+1) x dim_key)
        """
        batch_sizes = [len(episode_batch[key]) for key in episode_batch.keys()]
        assert np.all(np.array(batch_sizes) == batch_sizes[0])
        batch_size = batch_sizes[0]

        with self.lock:
            idxs = self._get_ordered_storage_idx(batch_size)  

            # load inputs into buffers
            for key in episode_batch.keys():
                if key in self.buffers:
                    self.buffers[key][idxs] = episode_batch[key]

            self.n_transitions_stored += batch_size * self.T


    def get_current_episode_size(self):
        with self.lock:
            return self.current_size

    def get_current_size(self):
        with self.lock:
            return self.current_size * self.T

    def get_transitions_stored(self):
        with self.lock:
            return self.n_transitions_stored

    def clear_buffer(self):
        with self.lock:
            self.current_size = 0
    
    # if full, insert in order
    def _get_ordered_storage_idx(self, inc=None):
        inc = inc or 1   # size increment
        assert inc <= self.size, "Batch committed to replay is too large!"

        if self.point+inc <= self.size - 1:
            idx = np.arange(self.point, self.point + inc)
        else:
            overflow = inc - (self.size - self.point)
            idx_a = np.arange(self.point, self.size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])

        self.point = (self.point + inc) % self.size

        # update replay size, don't add when it already surpass self.size
        if self.current_size < self.size:
            self.current_size = min(self.size, self.current_size+inc)

        self.training_size = self.current_size
        if inc == 1:
            idx = idx[0]
        return idx

    def save(self, path):
        save_buffer = {key: self.buffers[key][:self.current_size] for key in self.buffer_shapes.keys()}
        with open(path, "wb") as fp:  
            pickle.dump(save_buffer, fp)      
    
    def load(self, path):
        with open(path, "rb") as fp:  
            data = pickle.load(fp)     
            size = data['o'].shape[0]
            self.current_size = size
            # if size > self.size:
            self.buffers = {key: np.empty([size, *shape]) for key, shape in self.buffer_shapes.items()}
            self.size = size
            for key in self.buffer_shapes.keys():
                self.buffers[key][:size] = data[key][:size]


if __name__ == "__main__":
    buffer_shapes = {'a':(2, 1)}
    buffer = ReplayBuffer(buffer_shapes, 10, 2, None)
    buffer.store_episode({'a':np.random.random((1,2,1))})
    import pdb; pdb.set_trace()
