#! /bin/python3

import os, sys
from collections import OrderedDict
import torch
from glob import glob
import typer


def clean_ckpt(input_ckpt_filename):
    print(f' - Reading: {input_ckpt_filename}')
    output_ckpt_filename = input_ckpt_filename.replace('.ckpt', '') + '_LIGHT.ckpt'
    
    sd = torch.load(input_ckpt_filename, map_location='cpu')

    if 'optimizer_states' in sd.keys():
        print(' |-> Deleting: "optimizer_states"')
        del( sd['optimizer_states'] )

    if 'lr_schedulers' in sd.keys():
        print(' |-> Deleting: "lr_schedulers"')
        del( sd['lr_schedulers'] )

    if 'callbacks' in sd.keys():
        print(' |-> Deleting: "callbacks"')
        del( sd['callbacks'] )

    if 'native_amp_scaling_state' in sd.keys():
        print(' |-> Deleting: "native_amp_scaling_state"')
        del( sd['native_amp_scaling_state'] )

    print(' |-> Cleaning: "state_dict"')
    state_dict = OrderedDict()
    for k in sd['state_dict'].keys():
        if "criterion" not in k:
            state_dict[k] = sd['state_dict'][k]
            
        else:
            print(f'   |-> removing: "{k}"')

    sd['state_dict'] = state_dict
    
    print(f' |-> Writting output file: "{output_ckpt_filename}" ...', end='')
    torch.save(sd, output_ckpt_filename)
    print(' OK!!')

    print(40*' #')
    
    return None

def main(
        input_ckpt_filename=typer.Argument(
            "*.ckpt",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Path to checkpoint files",
        )
    ):
    
    input_ckpt_filename_v = glob(input_ckpt_filename)
    print(f'FOUND: {len(input_ckpt_filename_v)} checkpoints.')
    for input_ckpt_filename in input_ckpt_filename_v:
        clean_ckpt(input_ckpt_filename)
    
    return None


if __name__ == '__main__':
    typer.run(main)
