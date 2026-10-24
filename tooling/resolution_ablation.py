"""Controlled validation-only inference resolution comparison for one checkpoint."""
import hashlib
import argparse
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'.ultralytics'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path);args=parser.parse_args()
    import numpy as np
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4)
    plan=json.loads((ROOT/'resolution_plan.json').read_text())
    source=ROOT/plan['weights'];sha=hashlib.sha256(source.read_bytes()).hexdigest()
    reference=ROOT/'reports/dataset_reference.json'
    data=str(args.data or (Path(json.loads(reference.read_text())['data_yaml']) if reference.exists() else ROOT/'data/yolo/data.yaml'))
    out=ROOT/'reports/resolution_ablation';out.mkdir(exist_ok=True)
    results=[]
    for size in plan['candidate_resolutions']:
        saved=out/f'{size}.json'
        if saved.exists():
            result=json.loads(saved.read_text())
            if result['weights_sha256']!=sha:raise ValueError('Existing ablation uses different weights')
            results.append(result);continue
        model=YOLO(str(source))
        metrics=model.val(data=data,split='val',imgsz=size,batch=plan['batch'],device='0',workers=2,
            conf=plan['ap_confidence_floor'],iou=plan['nms_iou'],max_det=plan['max_det'],
            nms=None,plots=False,project=str(out),name=f'validation_{size}',exist_ok=False,verbose=False)
        box=metrics.box
        valid=np.flatnonzero(np.asarray(box.px)>=0.001)
        curve=np.asarray(box.f1_curve).mean(axis=0)
        index=int(valid[np.argmax(curve[valid])]);confidence=float(box.px[index])
        classes=[]
        for i,cid in enumerate(np.asarray(box.ap_class_index,dtype=int)):
            p=float(np.interp(confidence,box.px,box.p_curve[i]));r=float(np.interp(confidence,box.px,box.r_curve[i]))
            classes.append(dict(class_id=int(cid),class_name=model.names[int(cid)],
                precision=p,recall=r,f1=2*p*r/(p+r+1e-16),ap50=float(box.ap50[i]),ap50_95=float(box.ap[i])))
        result=dict(imgsz=size,weights_sha256=sha,split='val',batch=plan['batch'],
            mAP50=float(box.map50),mAP50_95=float(box.map),selected_confidence=confidence,
            macro_f1=float(np.mean([x['f1'] for x in classes])),classes=classes,
            gt_instances=int(np.asarray(metrics.nt_per_class).sum()))
        assert result['gt_instances']==4758 and len(classes)==18
        saved.write_text(json.dumps(result,indent=2),encoding='utf-8')
        results.append(result);print(json.dumps({k:result[k] for k in ('imgsz','mAP50','mAP50_95','macro_f1')}),flush=True)
        del model;torch.cuda.empty_cache()
    baseline=next(x for x in results if x['imgsz']==640)
    winner=max(results,key=lambda x:(x['mAP50_95'],-x['imgsz']))
    if winner['mAP50_95']-baseline['mAP50_95']<0.001:winner=baseline
    selection=dict(status='complete',selection_metric='validation mAP50-95',test_used_for_selection=False,
                   winner=winner,results=results,weights_sha256=sha)
    (out/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    print('Selected inference resolution',winner['imgsz'],flush=True)


if __name__=='__main__':main()
