# RF-DETR hardware pilot only
        RF-DETR was not trained on the full dataset and is not the submitted model.
        The pilot used148 training/74 validation images, no test images. It confirms
        GPU feasibility, not comparable detection quality. The user selected the
        verified YOLO26s submission instead of a further6–8hour full experiment.

        Optional reproduction, in a separate environment after the YOLO pipeline
        has generated trusted labels/train.cache and labels/val.cache:
        `pip install -r rfdetr_trial/requirements.txt`
        `python tooling/prepare_rfdetr_data.py --scope pilot --dataset-dir data/yolo`
        `python rfdetr_trial/probe_training.py --batch-size 8 --accum 2`
