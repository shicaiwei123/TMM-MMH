
import nw_ucla
import torch

def nw_ucla_datalaoder(args):

    nwucla_trainset = nw_ucla.NWUCLA_CS(data_root="/home/data/shicaiwei/N-UCLA/multiview_action",
                                        split_subject=args.split_subject,
                                        clip_len=args.clip_len,
                                        mode=args.mode,
                                        train=True,
                                        sample_interal=1)

    nwucla_testset = nw_ucla.NWUCLA_CS(data_root="/home/data/shicaiwei/N-UCLA/multiview_action",
                                       split_subject=args.split_subject,
                                       clip_len=args.clip_len,
                                       mode=args.mode,
                                       train=False,
                                       sample_interal=1)

    trainloader = torch.utils.data.DataLoader(nwucla_trainset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.workers,
                                              pin_memory=False)