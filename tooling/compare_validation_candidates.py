"""Compare all YOLO candidates per class on the same validation protocol, never test."""
import csv
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'.ultralytics'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path)
    parser.add_argument('--refine-weights',type=Path,default=ROOT/'runs/yolo11s_refine16/weights/best.pt')
    args=parser.parse_args()
    import numpy as np
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4)
    reference=ROOT/'reports/dataset_reference.json'
    data=str(args.data or (Path(json.loads(reference.read_text())['data_yaml']) if reference.exists() else ROOT/'data/yolo/data.yaml'))
    folder=ROOT/'reports/validation_candidates';folder.mkdir(exist_ok=True)
    baseline=ROOT/'models/public_yolo26n_reference.pt'
    if not baseline.exists():baseline=ROOT/'models/yolo26n_reference.pt'
    best=ROOT/'models/public_best.pt'
    if not best.exists():best=ROOT/'models/best.pt'
    candidates=[('YOLO26n',baseline,640,12),('YOLO26s',best,640,24),('YOLO11s_refine',args.refine_weights,768,28)]
    all_results=[];class_rows=[]
    for name,weights,size,epochs in candidates:
        if not weights.is_file():raise FileNotFoundError('Provide the model weights; use --refine-weights for the reproduced YOLO11s checkpoint')
        sha=hashlib.sha256(weights.read_bytes()).hexdigest();saved=folder/f'{name}.json'
        if saved.exists():
            result=json.loads(saved.read_text());assert result['weights_sha256']==sha
        else:
            model=YOLO(str(weights))
            metrics=model.val(data=data,split='val',imgsz=size,batch=8,workers=2,device='0',
                conf=0.001,iou=0.7,max_det=300,nms=None,plots=True,project=str(folder),name=name,
                exist_ok=False,verbose=False)
            box=metrics.box;eligible=np.flatnonzero(np.asarray(box.px)>=0.001)
            index=int(eligible[np.argmax(np.asarray(box.f1_curve).mean(axis=0)[eligible])]);confidence=float(box.px[index])
            classes=[]
            for i,cid in enumerate(np.asarray(box.ap_class_index,dtype=int)):
                p=float(np.interp(confidence,box.px,box.p_curve[i]));r=float(np.interp(confidence,box.px,box.r_curve[i]))
                classes.append(dict(class_id=int(cid),class_name=model.names[int(cid)],ap50=float(box.ap50[i]),
                    ap50_95=float(box.ap[i]),precision=p,recall=r,f1=2*p*r/(p+r+1e-16)))
            result=dict(name=name,weights_sha256=sha,imgsz=size,total_training_epochs=epochs,split='val',
                batch=8,selected_confidence=confidence,mAP50=float(box.map50),mAP50_95=float(box.map),
                macro_F1=float(np.mean([x['f1'] for x in classes])),classes=classes,
                note='Default confusion matrices use conf0.001/matching IoU0.45; operating matrix is a separate diagnostic.')
            saved.write_text(json.dumps(result,indent=2),encoding='utf-8')
            del model;gc.collect();torch.cuda.empty_cache()
        all_results.append(result)
        class_rows.extend(dict(model=name,**x) for x in result['classes'])
        print(name,result['mAP50'],result['mAP50_95'],result['macro_F1'],flush=True)
    with (folder/'per_class_comparison.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(class_rows[0]));writer.writeheader();writer.writerows(class_rows)
    results=dict(status='complete',split='val',test_used_for_selection=False,models=all_results,
                 winner=max(all_results,key=lambda x:x['mAP50_95'])['name'])
    (folder/'comparison.json').write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__=='__main__':main()
