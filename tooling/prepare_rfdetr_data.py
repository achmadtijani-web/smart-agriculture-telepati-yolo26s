"""Export the exact cleaned loader annotations to COCO; link immutable images on the SSD."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def select_samples(records,per_class,background):
    chosen={}
    ordered=sorted(records,key=lambda x:x['im_file'])
    for cid in range(18):
        matches=[x for x in ordered if cid in set(map(int,x['cls'].reshape(-1)))]
        for x in matches[:per_class]:chosen[x['im_file']]=x
    for x in [x for x in ordered if len(x['cls'])==0][:background]:chosen[x['im_file']]=x
    return list(chosen.values())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scope',choices=['pilot','full'],required=True);parser.add_argument('--dataset-dir',type=Path);args=parser.parse_args()
    reference=ROOT/'reports/dataset_reference.json'
    source=args.dataset_dir or (Path(json.loads(reference.read_text())['target']) if reference.exists() else ROOT/'data/yolo')
    output=(source.parent/f'telepati-rfdetr-{args.scope}-20261024').resolve()
    assert output.is_relative_to(source.parent) and output!=source
    classes=json.loads((ROOT/'reports/class_mapping.json').read_text())['classes']
    categories=[dict(id=x['yolo_id'],name=x['name'],supercategory='none') for x in classes]
    summary=dict(scope=args.scope,source=str(source),target=str(output),categories=categories,counts={},test_used_for_training=False)
    for split,destination in (('train','train'),('val','valid'),('test','test')):
        records=[] if args.scope=='pilot' and split=='test' else np.load(source/'labels'/f'{split}.cache',allow_pickle=True).item()['labels']
        if args.scope=='pilot':
            records=select_samples(records,8 if split=='train' else 4,4 if split=='train' else 2) if split!='test' else []
        folder=output/destination;folder.mkdir(parents=True,exist_ok=True)
        images=[];annotations=[];linked=0;copied=0
        for index,record in enumerate(records):
            image=Path(record['im_file']);target=folder/image.name
            if not target.exists():
                try:os.link(image,target);linked+=1
                except OSError:shutil.copy2(image,target);copied+=1
            h,w=record['shape'];images.append(dict(id=index,file_name=image.name,width=w,height=h))
            for cid,box in zip(record['cls'].reshape(-1),record['bboxes']):
                cid=int(cid);assert 0<=cid<18
                cx,cy,bw,bh=map(float,box);bbox=[(cx-bw/2)*w,(cy-bh/2)*h,bw*w,bh*h]
                annotations.append(dict(id=len(annotations)+1,image_id=index,category_id=cid,bbox=bbox,area=bbox[2]*bbox[3],iscrowd=0))
        payload=dict(info={'description':'Same cleaned Telepati18 classes; loader-deduplicated annotations'},licenses=[],images=images,annotations=annotations,categories=categories)
        encoded=json.dumps(payload,separators=(',',':')).encode()
        (folder/'_annotations.coco.json').write_bytes(encoded)
        summary['counts'][split]=dict(images=len(images),annotations=len(annotations),new_hardlinks=linked,new_copies=copied,annotation_sha256=hashlib.sha256(encoded).hexdigest())
        print(split,summary['counts'][split],flush=True)
    if args.scope=='full':
        assert [summary['counts'][s]['images'] for s in ('train','val','test')]==[8822,2046,1059]
        assert [summary['counts'][s]['annotations'] for s in ('train','val','test')]==[18002,4758,2669]
    destination=ROOT/'rfdetr_trial';destination.mkdir(exist_ok=True)
    (destination/f'data_{args.scope}.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')


if __name__=='__main__':main()
