import threading
import pickle
import numpy as np
from wgcsl.common import logger

class ReplayBuffer:
    def __init__(self, buffer_shapes, size_in_transitions, T, sample_transitions, default_sampler, info=None, validation_rate=0): 
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
        self.validation_size = 0
        self.T = T
        self.sample_transitions = sample_transitions
        self.default_sampler = default_sampler
        self.info = info
        self.kde = self.info['kde'] if 'kde' in self.info.keys() else None
        self.validation_rate = validation_rate
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

    def sample(self, batch_size, random=False, validation=False):
        """Returns a dict {key: array(batch_size x shapes[key])}
        """
        buffers = {}
        with self.lock:
            assert self.current_size > 0
            for key in self.buffers.keys():
                # buffers[key] = self.buffers[key][:self.current_size]
                if not validation:
                    buffers[key] = self.buffers[key][:self.training_size]
                else:
                    buffers[key] = self.buffers[key][self.training_size:]

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

        self.training_size = int((1 - self.validation_rate) * self.current_size)
        self.validation_size = self.current_size - self.training_size

        # ### mv trajs
        # traj_std = self.buffers['ag'].std(axis=1).sum(axis=1)
        # self.info['p'] = np.ones(self.size)
        # self.info['p'][traj_std < 0.0001] = 0

        # ## trajstd 
        # traj_std = self.buffers['ag'].std(axis=1).sum(axis=1)
        # self.info['p'] = self.buffers['ag'].std(axis=1).sum(axis=1)
        

        if self.kde:
            ags = self.buffers['ag'][:,:-1, :].reshape(-1, self.buffer_shapes['ag'][-1])
            M, T, N = self.buffers['g'].shape[0], self.buffers['g'].shape[1], self.buffers['g'].shape[2]
            relabel_pairs = np.zeros((M * T * (T+1)//2, N * 2))
            relabel_pairs_compact = np.zeros((M * T * (T+1)//2, N * 2))
            compact_num = 0
            full_2_compact_idxs = np.zeros(M * T * (T+1)//2)
            print('begin process kde ...')
            for i in range(M):
                start = i*T*(T+1)//2
                diff_points = []
                diff_points_idxs = []
                for j in range(T):
                    relabel_pairs[start: start + T-j, : N] = ags[i * T + j]
                    relabel_pairs[start: start + T-j, N: ] = ags[i * T + j:i * T + T]
                    start += T-j

                    ### for Fetch Tasks
                    if j == 0:
                        diff_points.append(ags[i * T + j])
                        diff_points_idxs.append(0)
                    elif len(diff_points):
                        if j >= 1:
                            diff = ags[i * T + j] - ags[i * T + j-1]
                            distance = np.abs(diff).sum()
                        if j == 0 or distance > 0.001:
                            diff_points.append(ags[i * T +j])
                            diff_points_idxs.append(j)
                L = len(diff_points)
                diff_points_trans = np.zeros(T)
                compact_start = compact_num
                for k in range(L):
                    relabel_pairs_compact[compact_num: compact_num + L-k, :N] = diff_points[k]
                    relabel_pairs_compact[compact_num: compact_num+L-k, N:] = diff_points[k:]
                    compact_num += L - k

                    p = diff_points_idxs[k]
                    q = diff_points_idxs[k+1] if k < L-1 else T
                    diff_points_trans[p:q] = p
                
                # compute full to compact array
                start = i*T*(T+1)//2
                for d in range(T):
                    for m in range(d,T):
                        d1, m1 = diff_points_trans[d], diff_points_trans[m]
                        d_diff, m_diff = diff_points_idxs.index(d1), diff_points_idxs.index(m1) 
                        full_2_compact_idxs[start + (2*T-d+1)*d //2 + m-d] = int(compact_start + (2*L- d_diff +1)*d_diff //2 + m_diff - d_diff)
                
            # print('fit KDE')
            # print(compact_num)
            # self.kde.fit(relabel_pairs) # [::5] for all data
            
            # path = '/home/ryangam/AWGCSL-master/offline_data/hard_tasks_2e6/expert/FetchPush/1000trajs_density/'
            # import os; os.mkdir(path)
            # self.kde.save(path)
            
            path = '/home/ryangam/AWGCSL-master/offline_data/hard_tasks_2e6/expert/FetchPush/density5/'
            self.kde.load(path+'KDE.pkl')
            self.kde.fitted_kde.atol = 0.0001
            # ## compute log density
            # relabel_pairs_compact = relabel_pairs_compact[:compact_num]
            # log_density = np.zeros(compact_num)
            # N = 20000
            # from tqdm import trange
            # for t in trange(compact_num // N):
            #     log_density[t*N:(t+1) * N] = self.kde.parrallel_score_samples(relabel_pairs_compact[t*N:(t+1) * N])
            # if compact_num // N *N < compact_num:
            #     log_density[compact_num // N *N : ] = self.kde.parrallel_score_samples(relabel_pairs_compact[compact_num // N *N:])

            # self.kde.save_log_density(log_density, path)
            log_density = self.kde.load_log_density(path)

            # norm_log_density = (log_density - log_density.mean()) / log_density.std()
            # density = np.clip(np.exp(norm_log_density), 0, 5)
            # weights = 1 / (density + 0.5)
            rank_idxs = log_density.argsort()[::-1]
            # rank_weight = 0.6 * (1 - np.arange(1, len(log_density)+1) / len(log_density)) + 0.7
            ### exp weight
            rank_weight = np.exp(1 - np.arange(1, len(log_density)+1) / len(log_density) - 0.5)
            logger.log('rank weight min:{}, 25:{}, medium:{}, 75:{} max:{}'.format(rank_weight[-1], 
                        rank_weight[3* len(log_density)//4], rank_weight[ len(log_density)//2], 
                        rank_weight[1* len(log_density)//4], rank_weight[0]))

            # rank-based weight
            self.info['kde_weights_compact'] = rank_weight[rank_idxs]
            self.info['kde_weights_trans'] = full_2_compact_idxs
            # import pdb;pdb.set_trace()

if __name__ == "__main__":
    buffer_shapes = {'a':(2, 1)}
    buffer = ReplayBuffer(buffer_shapes, 10, 2, None)
    buffer.store_episode({'a':np.random.random((1,2,1))})
    import pdb; pdb.set_trace()
