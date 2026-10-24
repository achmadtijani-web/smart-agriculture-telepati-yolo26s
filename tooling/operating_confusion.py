"""Validation-only confusion matrix at the F1-selected operating confidence."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'.ultralytics'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--weights',type=Path,default=ROOT/'models/public_best.pt')
    parser.add_argument('--metrics',type=Path,default=ROOT/'reports/submission_evaluation/metrics.json')
    parser.add_argument('--output',type=Path,default=ROOT/'reports/diagnostics_operating_val')
    parser.add_argument('--dataset-dir',type=Path)
    args=parser.parse_args()
    metrics=json.loads(args.metrics.read_text(encoding='utf-8-sig'))
    sha=hashlib.sha256(args.weights.read_bytes()).hexdigest()
    if sha!=metrics['weights_sha256']:raise ValueError('Weights do not match evaluated checkpoint')
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('Use a new output folder')
    if not args.output.resolve().is_relative_to(ROOT):raise ValueError('Output must be inside this project')
    import numpy as np
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.metrics import ConfusionMatrix
    torch.set_num_threads(4)
    model=YOLO(str(args.weights));names=model.names
    mapping=json.loads((ROOT/'reports/class_mapping.json').read_text(encoding='utf-8'))
    assert names=={x['yolo_id']:x['name'] for x in mapping['classes']}
    reference=ROOT/'reports/dataset_reference.json'
    dataset=args.dataset_dir or (Path(json.loads(reference.read_text(encoding='utf-8'))['target']) if reference.exists() else ROOT/'data/yolo')
    matrix=ConfusionMatrix(names=names)
    confidence=metrics['selected_confidence'];queue=[]
    stream=model.predict(source=str(dataset/'images/val'),stream=True,batch=8,
        imgsz=metrics['imgsz'],conf=confidence,iou=metrics['nms_iou'],max_det=metrics['max_det'],
        nms=None,device='0',verbose=False,save=False)
    for index,result in enumerate(stream):
        image=Path(result.path);h,w=result.orig_shape
        label=dataset/'labels/val'/f'{image.stem}.txt'
        unique=sorted({tuple(map(float,line.split())) for line in label.read_text().splitlines() if line.strip()})
        cls=[];boxes=[]
        for cid,cx,cy,bw,bh in unique:
            cls.append(cid);boxes.append([(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h])
        batch={'cls':torch.tensor(cls),'bboxes':torch.tensor(boxes,dtype=torch.float32).reshape(-1,4)}
        predicted={'cls':result.boxes.cls.cpu(),'conf':result.boxes.conf.cpu(),'bboxes':result.boxes.xyxy.cpu()}
        before=matrix.matrix.copy()
        matrix.process_batch(predicted,batch,conf=confidence,iou_thres=0.50)
        delta=matrix.matrix-before
        wrong=int(delta[:18,:18].sum()-np.trace(delta[:18,:18]));fn=int(delta[18,:18].sum());fp=int(delta[:18,18].sum())
        queue.append(dict(image=image.name,gt_instances=len(unique),predicted_instances=len(result.boxes),
                          class_confusions=wrong,missed_boxes=fn,unmatched_predictions=fp,error_events=wrong+fn+fp))
        if (index+1)%500==0:print('Validation images processed:',index+1,flush=True)
    args.output.mkdir(parents=True)
    labels=[f'{cid}: {names[cid]}' for cid in range(18)]+['background']
    with (args.output/'confusion_matrix_counts.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['predicted / actual',*labels])
        writer.writerows([name,*map(int,row)] for name,row in zip(labels,matrix.matrix))
    queue.sort(key=lambda row:row['error_events'],reverse=True)
    with (args.output/'review_queue.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(queue[0]));writer.writeheader();writer.writerows(queue)
    pairs=[]
    for p in range(18):
        for t in range(18):
            if p!=t and matrix.matrix[p,t]:
                pairs.append(dict(actual_class=names[t],predicted_class=names[p],count=int(matrix.matrix[p,t])))
    pairs.sort(key=lambda row:row['count'],reverse=True)
    assert int(matrix.matrix[:,:18].sum())==metrics['val']['gt_instances']==4758
    assert len(queue)==2046
    for normalized in (False,True):matrix.plot(normalize=normalized,save_dir=args.output)
    summary=dict(status='complete',split='val',images=2046,gt_instances=4758,weights_sha256=sha,
        confidence=confidence,matching_iou=0.50,nms_iou=metrics['nms_iou'],imgsz=metrics['imgsz'],
        axes='rows=predicted, columns=actual; final row/column is background',top_class_confusions=pairs[:20],
        unmatched_ground_truth=int(matrix.matrix[18,:18].sum()),unmatched_predictions=int(matrix.matrix[:18,18].sum()),
        correct_class_and_box_matches=int(np.trace(matrix.matrix[:18,:18])),
        note='Diagnostic class-agnostic IoU-greedy confusion assignment differs from class-aware AP curve matching. Do not replace submitted macro-F1 by arithmetic from this matrix.')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
