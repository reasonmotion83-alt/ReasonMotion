import numpy as np
import os
from torch.utils.data import Dataset
from skeleton import Skeleton

# AMASS dataset seq shape: (150, 22, 3)

class DatasetAMASS(Dataset):

    def __init__(self, mode, t_his=30, t_pred=120, use_vel=False):
        super().__init__()
        self.use_vel = use_vel
        self.mode = mode
        if mode == 'train':
            self.data_file = os.path.join('/home/kingjames23/datasets/amass', 'data_3d_amass.npz')
        elif mode == 'test':
            self.data_file = os.path.join('/home/kingjames23/datasets/amass', 'data_3d_amass_test.npz')
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        self.t_his = t_his
        self.t_pred = t_pred
        self.t_total = t_his + t_pred
        self.use_vel = use_vel
        self.prepare_data()

    def prepare_data(self):
        self.skeleton = Skeleton(parents=[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19],
                                 joints_left=[1, 4, 7, 10, 13, 16, 18, 20],
                                 joints_right=[2, 5, 8, 11, 14, 17, 19, 21])
        self.kept_joints = np.arange(22)
        self.process_data()

    def process_data(self):
        data_o = np.load(self.data_file, allow_pickle=True)
        data_f = data_o['arr_0']  # (#samples, 150, 22, 3)
        if self.use_vel:
            raise NotImplementedError
        self.data = data_f

    def sample(self):
        n_samples = self.data.shape[0]
        idx = np.random.randint(0, n_samples)
        traj = self.data[idx]
        return traj[None, ...]

    def sampling_generator(self, num_samples=1000, batch_size=8, aug=True):
        if self.mode != 'train':
            aug = False
        for i in range(num_samples // batch_size):
            sample = []
            for i in range(batch_size):
                sample_i = self.sample()
                sample.append(sample_i)
            sample = np.concatenate(sample, axis=0)
            if aug is True:
                if np.random.uniform() > 0.5:  # x-y rotating
                    theta = np.random.uniform(0, 2 * np.pi)
                    rotate_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                    rotate_xy = np.matmul(sample.transpose([0, 2, 1, 3])[..., 0:2], rotate_matrix)
                    sample[..., 0:2] = rotate_xy.transpose([0, 2, 1, 3])
                    del theta, rotate_matrix, rotate_xy
                if np.random.uniform() > 0.5:  # x-z mirroring
                    sample[..., 0] = - sample[..., 0]
                if np.random.uniform() > 0.5:  # y-z mirroring
                    sample[..., 1] = - sample[..., 1]
            yield sample

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        traj = self.data[idx].copy()  # (T, 22, 3)

        # if traj.shape[0] < self.t_total:
        #     pad = self.t_total - traj.shape[0]
        #     traj = np.concatenate([traj, np.repeat(traj[-1:], pad, axis=0)], axis=0)
        # traj = traj[:self.t_total]

        if self.mode == 'train':
            if np.random.uniform() > 0.5:  # x-y rotating
                theta = np.random.uniform(0, 2 * np.pi)
                rotate_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                traj[..., 0:2] = np.matmul(traj[..., 0:2], rotate_matrix)
            if np.random.uniform() > 0.5:  # x-z mirroring
                traj[..., 0] = -traj[..., 0]
            if np.random.uniform() > 0.5:  # y-z mirroring
                traj[..., 1] = -traj[..., 1]

        pose = traj.reshape(self.t_total, -1).astype(np.float32)
        mask = np.zeros((self.t_total, pose.shape[1]), dtype=np.float32)
        mask[:self.t_his] = 1.0

        return {
            'pose': pose,
            'mask': mask,
            'timepoints': np.arange(self.t_total),
            'motion_name': None,
            'judge_score': None,
        }

    def iter_generator(self, step=None):
        num_samples = self.data.shape[0]
        for i in range(num_samples):
            seq = self.data[i]
            yield seq[None, ...]

    def sample_iter_action(self, ds_category, dataset_type='amass'):
        sample = [] 

        # ['Transitions', 'SSM', 'DFaust', 'DanceDB', 'GRAB', 'HUMAN4D', 'SOMA']
        if ds_category == 'Transitions':
            i0, i = [0, 233]
        elif ds_category == 'SSM':
            i0, i = [233, 245]
        elif ds_category == 'DFaust':
            i0, i = [245, 342]
        elif ds_category == 'DanceDB':
            i0, i = [342, 6321]
        elif ds_category == 'GRAB':
            i0, i = [6321, 10437]
        elif ds_category == 'HUMAN4D':
            i0, i = [10437, 12317]
        elif ds_category == 'SOMA':
            i0, i = [12317, 12727]
        else:
            raise

        idx = np.random.randint(i0, i)
        traj = self.data[idx]
        sample.append(traj[None, ...])

        sample = np.concatenate(sample, axis=0)
        return sample


if __name__ == '__main__':
    np.random.seed(0)

    def check_raw_data_shapes(dataset):
        print("=" * 60)
        print("[1] Checking raw self.data shapes")

        shapes = [x.shape for x in dataset.data]
        unique_shapes = sorted(set(shapes))
        print(f"Number of samples: {len(shapes)}")
        print(f"Number of unique shapes: {len(unique_shapes)}")

        if len(unique_shapes) <= 10:
            print("Unique shapes:")
            for s in unique_shapes:
                print(f"  {s}")
        else:
            print("Too many unique shapes, showing first 10 only:")
            for s in unique_shapes[:10]:
                print(f"  {s}")

        lengths = [x.shape[0] for x in dataset.data]
        joint_dims = [x.shape[1:] for x in dataset.data]

        print(f"Min sequence length: {min(lengths)}")
        print(f"Max sequence length: {max(lengths)}")
        print(f"Unique joint dims: {sorted(set(joint_dims))}")

        shorter = [i for i, x in enumerate(dataset.data) if x.shape[0] < dataset.t_total]
        print(f"Samples shorter than t_total ({dataset.t_total}): {len(shorter)}")

        if len(shorter) > 0:
            print("First 10 short sample indices and shapes:")
            for i in shorter[:10]:
                print(f"  idx={i}, shape={dataset.data[i].shape}")

    def check_getitem_output(dataset, num_checks=5):
        print("=" * 60)
        print("[2] Checking __getitem__ outputs")

        indices = np.random.choice(len(dataset), size=min(num_checks, len(dataset)), replace=False)
        for idx in indices:
            item = dataset[idx]
            pose = item['pose']
            mask = item['mask']
            timepoints = item['timepoints']

            print(f"idx={idx}")
            print(f"  raw traj shape: {dataset.data[idx].shape}")
            print(f"  pose shape: {pose.shape}")
            print(f"  mask shape: {mask.shape}")
            print(f"  timepoints shape: {timepoints.shape}")
            print(f"  motion_name: {item['motion_name']}")
            print(f"  judge_score: {item['judge_score']}")

            assert pose.shape[0] == dataset.t_total, \
                f"pose first dim mismatch: got {pose.shape[0]}, expected {dataset.t_total}"
            assert mask.shape == pose.shape, \
                f"mask shape {mask.shape} != pose shape {pose.shape}"
            assert timepoints.shape[0] == dataset.t_total, \
                f"timepoints length mismatch: got {timepoints.shape[0]}, expected {dataset.t_total}"

            expected_feat_dim = 22 * 3
            assert pose.shape[1] == expected_feat_dim, \
                f"pose second dim mismatch: got {pose.shape[1]}, expected {expected_feat_dim}"

        print("getitem check passed.")

    def check_sampling_generator(dataset, num_batches=2, batch_size=4):
        print("=" * 60)
        print("[3] Checking sampling_generator outputs")

        gen = dataset.sampling_generator(
            num_samples=num_batches * batch_size,
            batch_size=batch_size,
            aug=False
        )

        for b_idx, batch in enumerate(gen):
            print(f"batch {b_idx}: shape = {batch.shape}")
            assert batch.shape[0] == batch_size, \
                f"batch size mismatch: got {batch.shape[0]}, expected {batch_size}"
            assert batch.shape[1] == dataset.t_total, \
                f"time dim mismatch: got {batch.shape[1]}, expected {dataset.t_total}"
            assert batch.shape[2:] == (22, 3), \
                f"joint/dim mismatch: got {batch.shape[2:]}, expected (22, 3)"

        print("sampling_generator check passed.")

    def compare_raw_vs_getitem(dataset, idx=0):
        print("=" * 60)
        print("[4] Comparing raw sample and __getitem__ output")

        raw = dataset.data[idx]
        item = dataset[idx]
        pose = item['pose'].reshape(dataset.t_total, 22, 3)

        print(f"raw shape: {raw.shape}")
        print(f"getitem pose reshaped: {pose.shape}")

        usable_len = min(raw.shape[0], dataset.t_total)
        same_prefix = np.allclose(raw[:usable_len], pose[:usable_len], atol=1e-6)
        print(f"Prefix equal for first {usable_len} frames: {same_prefix}")

        if raw.shape[0] < dataset.t_total:
            print("WARNING: raw sequence is shorter than t_total.")
            print("Since padding is currently commented out, __getitem__ may fail or return wrong shape.")
        else:
            print("raw length is enough for truncation-only pipeline.")

    dataset = DatasetAMASS('train')

    check_raw_data_shapes(dataset)
    check_getitem_output(dataset, num_checks=5)
    check_sampling_generator(dataset, num_batches=2, batch_size=4)
    compare_raw_vs_getitem(dataset, idx=0)