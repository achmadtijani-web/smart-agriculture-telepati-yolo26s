"""Hardware pilot only:148 train/74 validation images; no test and no final accuracy claim."""
import json
import argparse
import os
from pathlib import Path
import statistics
import time

ROOT=Path(__file__).resolve().parents[1]
TRIAL=ROOT/'rfdetr_trial'
os.environ.setdefault('RF_HOME',str(TRIAL/'.cache'))
os.environ.setdefault('HF_HOME',str(TRIAL/'.cache/huggingface'))
os.environ.setdefault('TORCH_HOME',str(TRIAL/'.cache/torch'))
os.environ.setdefault('MPLCONFIGDIR',str(TRIAL/'.cache/matplotlib'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--batch-size',type=int,default=1);parser.add_argument('--accum',type=int,default=16)
    args=parser.parse_args();assert args.batch_size>0 and args.accum>0
    import torch
    from pytorch_lightning import Callback, seed_everything
    from rfdetr.config import RFDETRNanoConfig,TrainConfig
    from rfdetr.training import RFDETRModelModule,RFDETRDataModule,build_trainer
    torch.set_num_threads(4);seed_everything(42,workers=True)
    data=json.loads((TRIAL/'data_pilot.json').read_text())
    names=[x['name'] for x in sorted(data['categories'],key=lambda x:x['id'])]
    assert len(names)==18
    output=TRIAL/f'pilot_640_b{args.batch_size}_g{args.accum}'
    output.mkdir(exist_ok=True)
    report=TRIAL/f'pilot_status_b{args.batch_size}_g{args.accum}.json'
    def write(value):report.write_text(json.dumps(value,indent=2,default=str),encoding='utf-8')
    write(dict(status='starting',phase='hardware_feasibility_only',test_used=False,train_images=148,val_images=74))
    class Timing(Callback):
        def __init__(self):self.durations=[];self.cadence=[];self.last=None
        def on_train_batch_start(self,trainer,module,batch,batch_idx):
            torch.cuda.synchronize();now=time.perf_counter()
            if self.last is not None and batch_idx>warmup:self.cadence.append(now-self.last)
            self.last=now;self.begin=now
        def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
            torch.cuda.synchronize()
            if batch_idx>warmup:self.durations.append(time.perf_counter()-self.begin)
    warmup=min(16,max(2,148//args.batch_size//4))
    timer=Timing()
    try:
        mc=RFDETRNanoConfig(num_classes=18,resolution=640,positional_encoding_size=40,
                           gradient_checkpointing=True,device='cuda')
        tc=TrainConfig(dataset_dir=data['target'],dataset_file='roboflow',output_dir=str(output),
            epochs=1,batch_size=args.batch_size,grad_accum_steps=args.accum,lr=1e-4,lr_encoder=1.5e-4,
            num_workers=0,multi_scale=False,expanded_scales=False,scale_jitter=False,
            use_ema=True,eval_ema_only=True,eval_max_dets=300,tensorboard=False,wandb=False,
            run_test=False,progress_bar=None,class_names=names,seed=42,
            notes={'purpose':'hardware pilot only; not a candidate trained on full data'})
        (output/'requested_config.json').write_text(json.dumps(dict(model=mc.model_dump(mode='json'),train=tc.model_dump(mode='json')),indent=2),encoding='utf-8')
        started=time.perf_counter();module=RFDETRModelModule(mc,tc);dm=RFDETRDataModule(mc,tc)
        trainer=build_trainer(tc,mc,accelerator='gpu',devices=1,num_sanity_val_steps=0)
        trainer.callbacks.append(timer)
        torch.cuda.reset_peak_memory_stats();trainer.fit(module,dm)
        seconds=time.perf_counter()-started
        cadence=statistics.mean(timer.cadence)
        result=dict(status='complete',scope='Hardware pilot only;148 training/74 validation images and no test images.',
            batch_size=args.batch_size,grad_accum_steps=args.accum,resolution=640,gradient_checkpointing=True,
            step_seconds_after_warmup=statistics.mean(timer.durations),batch_cadence_seconds=cadence,
            estimated_full_training_epoch_minutes=8822*cadence/args.batch_size/60,
            estimated_12_epoch_training_hours=12*8822*cadence/args.batch_size/3600,
            note='Estimates exclude full2046-image validation and checkpoint I/O; pilot accuracy is not compared to full-data YOLO.',
            elapsed_seconds=seconds,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
            peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,gpu=torch.cuda.get_device_name(0),
            rfdetr='1.9.0',torch=torch.__version__)
        write(result);print(json.dumps(result,indent=2),flush=True)
    except Exception as exc:
        write(dict(status='failed',error=f'{type(exc).__name__}: {exc}',scope='hardware pilot; no model comparison claim'))
        raise


if __name__=='__main__':main()
