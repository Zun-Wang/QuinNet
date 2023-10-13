import torch
from torch_geometric.data import InMemoryDataset, download_url, Data
import tarfile
from pytorch_lightning.utilities import rank_zero_warn
import numpy as np
import os
from tqdm import tqdm


class rMD17(InMemoryDataset):
    molecule_files = dict(
        aspirin='rmd17_aspirin.npz',
        azobenzene='rmd17_azobenzene.npz',
        benzene='rmd17_benzene.npz',
        ethanol='rmd17_ethanol.npz',
        malonaldehyde='rmd17_malonaldehyde.npz',
        naphthalene='rmd17_naphthalene.npz',
        paracetamol='rmd17_paracetamol.npz',
        salicylic='rmd17_salicylic.npz',
        toluene='rmd17_toluene.npz',
        uracil='rmd17_uracil.npz',
    )

    available_molecules = list(molecule_files.keys())

    def __init__(self, root, transform=None, pre_transform=None, dataset_arg=None):
        assert dataset_arg is not None, (
            "Please provide the desired comma separated molecule(s) through"
            f"'dataset_arg'. Available molecules are {', '.join(rMD17.available_molecules)} "
            "or 'all' to train on the combined dataset."
        )

        if dataset_arg == 'all':
            dataset_arg = ','.join(rMD17.available_molecules)

        self.molecules = dataset_arg.split(',')

        if len(self.molecules) > 1:
            rank_zero_warn('MD17 molecules have different reference energies, which is not accounted for during training.')

        super(rMD17, self).__init__(root, transform, pre_transform)

        self.offsets = [0]
        self.data_all, self.slices_all = [], []
        for path in self.processed_paths:
            data, slices = torch.load(path)
            self.data_all.append(data)
            self.slices_all.append(slices)
            self.offsets.append(len(slices[list(slices.keys())[0]]) - 1 + self.offsets[-1])

    def len(self):
        return sum(len(slices[list(slices.keys())[0]]) - 1 for slices in self.slices_all)

    def get(self, idx):
        data_idx = 0
        while data_idx < len(self.data_all) - 1 and idx >= self.offsets[data_idx + 1]:
            data_idx += 1

        self.data = self.data_all[data_idx]
        self.slices = self.slices_all[data_idx]
        return super(rMD17, self).get(idx - self.offsets[data_idx])

    @property
    def raw_file_names(self):
        return ['rmd17/npz_data/' + rMD17.molecule_files[mol] for mol in self.molecules]

    @property
    def processed_file_names(self):
        return [f'rmd17-{mol}.pt' for mol in self.molecules]

    def download(self):
        if not os.path.exists(os.path.join(self.raw_dir, 'rmd17.tar.bz2')):
            download_url('https://figshare.com/ndownloader/files/23950376', self.raw_dir)
            os.rename(os.path.join(self.raw_dir, '23950376'), os.path.join(self.raw_dir, 'rmd17.tar.bz2'))

        tar = tarfile.open(os.path.join(self.raw_dir, 'rmd17.tar.bz2'), 'r:bz2')
        tar.extractall(self.raw_dir)
        tar.close()


    def process(self):
        for path, processed_path in zip(self.raw_paths, self.processed_paths):
            data_npz = np.load(path)
            z = torch.from_numpy(data_npz['nuclear_charges']).long()
            positions = torch.from_numpy(data_npz['coords']).float()
            energies = torch.from_numpy(data_npz['energies']).float()
            forces = torch.from_numpy(data_npz['forces']).float()
            energies.unsqueeze_(1)

            samples = []
            for pos, y, dy in tqdm(zip(positions, energies, forces), total=energies.size(0)):
                
                data = Data(z=z, pos=pos, y=y.unsqueeze(1), dy=dy)

                if self.pre_filter is not None:
                    data = self.pre_filter(data)

                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                    
                samples.append(data)

            data, slices = self.collate(samples)
            torch.save((data, slices), processed_path)

    def get_atomref(self, max_z=100):
        """Refer to https://github.com/atomistic-machine-learning/schnetpack/blob/master/src/schnetpack/datasets/md17.py#L244-L256
        """
        refs = torch.zeros(max_z)
        refs[1] = -313.5150902000774  # H
        refs[6] = -23622.587180094913  # C
        refs[7] = -34219.46811826416  # N
        refs[8] = -47069.30768969713  # O
        return refs.view(-1, 1)