"""Remove machine-specific metadata; prove all model tensors remain identical."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PureWindowsPath
import re


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensors_digest(checkpoint):
    import torch
    digest = hashlib.sha256()
    for key in ('model', 'ema'):
        model = checkpoint.get(key)
        if model is None:
            continue
        for name, tensor in sorted(model.state_dict().items()):
            digest.update(f'{key}/{name}:{tensor.dtype}:{tuple(tensor.shape)}'.encode())
            digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def portable_checkpoint(source, output, report):
    import torch
    torch.set_num_threads(2)
    source, output, report = map(lambda p:Path(p).resolve(), (source,output,report))
    if output == source or output.exists():
        raise ValueError('Use a new output file; do not overwrite original or prior output')
    checkpoint = torch.load(source,map_location='cpu',weights_only=False)
    before = tensors_digest(checkpoint)
    visited = set()
    replaced = 0

    def clean(value, key=''):
        nonlocal replaced
        if isinstance(value,str) and re.match(r'^(?:[A-Za-z]:[\\/]|\\\\)',value):
            replaced += 1
            if key == 'data':
                return 'data/yolo/data.yaml'
            if key == 'project':
                return 'runs'
            if key == 'save_dir':
                return 'runs/' + PureWindowsPath(value).name
            return PureWindowsPath(value).name
        if id(value) in visited:
            return value
        if isinstance(value,(dict,list,tuple,torch.nn.Module)):
            visited.add(id(value))
        if isinstance(value,dict):
            for child_key in list(value):
                value[child_key] = clean(value[child_key],str(child_key))
        elif isinstance(value,list):
            for index,item in enumerate(value):
                value[index] = clean(item,key)
        elif isinstance(value,tuple):
            return tuple(clean(item,key) for item in value)
        elif isinstance(value,torch.nn.Module):
            for child_key,item in list(vars(value).items()):
                setattr(value,child_key,clean(item,child_key))
        return value

    clean(checkpoint)
    if tensors_digest(checkpoint) != before:
        raise RuntimeError('Tensor state changed while cleaning metadata')
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(checkpoint,output)
    loaded = torch.load(output,map_location='cpu',weights_only=False)
    after = tensors_digest(loaded)
    if before != after:
        raise RuntimeError('Saved tensor state differs')
    result = dict(status='pass',source_checkpoint_sha256=sha(source),published_checkpoint_sha256=sha(output),
                  tensor_state_sha256=after,all_tensors_unchanged=True,absolute_metadata_paths_replaced=replaced,
                  note='Only machine-specific metadata paths changed. Public weights are evaluated again before packaging.')
    report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--report',required=True,type=Path)
    args = parser.parse_args()
    print(json.dumps(portable_checkpoint(args.source,args.output,args.report),indent=2))
