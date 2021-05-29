#! /user/bin/env python

import tarfile
from torch.utils.data import Dataset

class QM9BZ2Dataset(Dataset):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.fp = tarfile.open(self.filename, "r:bz2")
        # filter out .xyz file
        self.file_info = [fn for fn in self.fp if fn.name.endswith(".xyz")]

    def __len__(self):
        return len(self.file_info)

    def __getitem__(self, idx):
        info = self.file_info[idx]
        fin = self.fp.extractfile(info)
        return fin.read()

if __name__ == "__main__":
    qmds = QM9BZ2Dataset("./datafolder/dsgdb9nsd.xyz.tar.bz2")
    print(qmds[12])