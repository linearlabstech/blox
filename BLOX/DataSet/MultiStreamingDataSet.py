
from .DataSet import DataSet
from BLOX.Common.utils import StrOrDict as sod
class MultiStreamingDataSet:

    def __init__(self,cfg=None):
        self.idx = 0
        self.size = -1
        self.ds_idx = -1
        self.n_data_sets = 0
        self.data_sets = []
        self.curr_ds = None
        self.mode = 'cpu'
        if cfg:self.init(cfg)
    
    def __len__(self):return self.size

    def cpu(self):
        self.mode = 'cpu'
        self.curr_ds.cpu()
        return self
    
    def cuda(self):
        self.mode = 'cuda'
        self.curr_ds.cuda()
        return self

    def __getitem__(self,idx):

        itr = idx % int(sum( self.sizes[:self.ds_idx]  ) )
        return self.curr_ds[itr]

    def load(self,ds):
        self.curr_ds = getattr(DataSet(ds['file']),self.mode)()
        self.ds_idx += 1

    def reset(self):



    def init(self,cfg):
        cfg = sod(cfg)
        self.data_sets = cfg['DataSets']
        self.n_data_sets = len(self.data_sets)
        self.sizes = [ d['size'] for d in self.data_sets]
        self.size = int(sum(self.sizes))

