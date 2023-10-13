import torch
from torch_geometric.data import InMemoryDataset, download_url, Data
from pytorch_lightning.utilities import rank_zero_warn
import numpy as np
from tqdm import tqdm


class MD22(InMemoryDataset):
    raw_url = 'http://www.quantum-machine.org/gdml/repo/datasets/'

    molecule_files = dict(
        Ac_Ala3_NHMe='md22_Ac-Ala3-NHMe.npz', 
        Docosahexaenoic_acid='md22_DHA.npz', 
        Stachyose='md22_stachyose.npz', 
        AT_AT='md22_AT-AT.npz', 
        AT_AT_CG_CG='md22_AT-AT-CG-CG.npz', 
        Buckyball_catcher='md22_buckyball-catcher.npz', 
        Double_walled_nanotube='md22_double-walled_nanotube.npz'
    )

    available_molecules = list(molecule_files.keys())

    def __init__(self, root, transform=None, pre_transform=None, dataset_arg=None):
        assert dataset_arg is not None, (
            "Please provide the desired comma separated molecule(s) through"
            f"'dataset_arg'. Available molecules are {', '.join(MD22.available_molecules)} "
            "or 'all' to train on the combined dataset."
        )

        if dataset_arg == 'all':
            dataset_arg = ','.join(MD22.available_molecules)

        self.molecules = dataset_arg.split(',')

        if len(self.molecules) > 1:
            rank_zero_warn('MD17 molecules have different reference energies, which is not accounted for during training.')

        super(MD22, self).__init__(root, transform, pre_transform)

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
        return super(MD22, self).get(idx - self.offsets[data_idx])

    @property
    def raw_file_names(self):
        return [MD22.molecule_files[mol] for mol in self.molecules]

    @property
    def processed_file_names(self):
        return [f'md22-{mol}.pt' for mol in self.molecules]

    def download(self):
        for file_name in self.raw_file_names:
            download_url(MD22.raw_url + file_name, self.raw_dir)

    def process(self):
        for path, processed_path in zip(self.raw_paths, self.processed_paths):
            data_npz = np.load(path)
            z = torch.from_numpy(data_npz['z']).long()
            positions = torch.from_numpy(data_npz['R']).float()
            energies = torch.from_numpy(data_npz['E'].reshape(-1)).float()
            forces = torch.from_numpy(data_npz['F']).float()

            samples = []
            for pos, y, dy in tqdm(zip(positions, energies, forces), total=energies.size(0)):
                
                data = Data(z=z, pos=pos, y=y.unsqueeze(-1), dy=dy)

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
