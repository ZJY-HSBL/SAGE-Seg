# Quickstart / 快速开始

1. Install dependencies / 安装依赖：`pip install -r requirements.txt`
2. Prepare dataset directories / 准备数据集目录。
3. Generate labeled/unlabeled lists / 生成有标记与无标记样本列表：`python tools/prepare_splits.py ...`
4. Edit YAML paths / 修改 YAML 路径。
5. Train / 训练：`python tools/train.py --config configs/voc.yaml`
6. Evaluate / 评估：`python tools/evaluate.py --config configs/voc.yaml --checkpoint runs/SAGE-Seg-VOC/best.pt`
7. Inference / 推理：`python tools/infer.py --config configs/voc.yaml --checkpoint runs/SAGE-Seg-VOC/best.pt --input image.jpg --output mask.png`
