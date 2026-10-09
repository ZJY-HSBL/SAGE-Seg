# SAGE-Seg

**Semantic-aware gated enhancement for semi-supervised segmentation · 面向半监督语义分割的语义感知门控增强框架**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-black.svg)](LICENSE)

SAGE-Seg is a compact semi-supervised semantic segmentation framework built around a teacher-student EMA architecture. It improves the treatment of unlabeled data at three levels: pixel-level semantic consistency, region-level sparse filtering, and sample-level confidence-adaptive augmentation routing.

SAGE-Seg 是一个基于教师-学生 EMA 架构的半监督语义分割框架。它从三个层面优化未标记数据的使用方式：像素级语义一致性、区域级稀疏过滤，以及样本级置信度自适应增强路由。

> Repository presentation is intentionally project-oriented. Method provenance and scope notes are retained in [`NOTICE.md`](NOTICE.md).

## Core design · 核心设计

- **SCD — Semantic Consistency Discrimination / 语义一致性判别**: compares teacher and student probability vectors at every pixel using cosine similarity and locates semantically inconsistent regions.
- **SRF — Sparse Region Filtering / 稀疏区域过滤**: applies 4-connected component analysis to the inconsistent mask and removes small fragmented regions before semantic mixing.
- **Confidence Gate / 置信度门控**: estimates sample confidence with normalized entropy and samples a Bernoulli gate to choose between strong augmentation and semantic mixing.
- **EMA Teacher / EMA 教师模型**: the student is optimized by gradient descent while the teacher is updated with exponential moving average parameters.
- **DeepLabV3+ / ResNet-101**: included as the default segmentation model, with dataset-specific training presets for PASCAL VOC 2012 and Cityscapes.

## Repository structure · 仓库结构

```text
SAGE-Seg/
├── README.md
├── LICENSE
├── NOTICE.md
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── voc.yaml
│   └── cityscapes.yaml
├── sage_seg/
│   ├── augment/
│   │   └── weak_strong.py
│   ├── core/
│   │   ├── gating.py
│   │   ├── mixing.py
│   │   ├── scd.py
│   │   └── srf.py
│   ├── data/
│   │   ├── cityscapes.py
│   │   ├── common.py
│   │   ├── factory.py
│   │   └── voc.py
│   ├── engine/
│   │   ├── evaluator.py
│   │   └── trainer.py
│   ├── losses/
│   │   └── segmentation.py
│   ├── metrics/
│   │   └── miou.py
│   ├── models/
│   │   ├── deeplabv3plus.py
│   │   └── ema.py
│   └── utils/
│       ├── checkpoint.py
│       ├── config.py
│       ├── image.py
│       └── seed.py
├── tools/
│   ├── evaluate.py
│   ├── infer.py
│   ├── prepare_splits.py
│   └── train.py
├── scripts/
│   ├── train_voc.sh
│   ├── train_cityscapes.sh
│   └── train_voc.ps1
└── tests/
    ├── test_gating.py
    ├── test_mixing.py
    ├── test_scd.py
    └── test_srf.py
```

## Installation · 安装

```bash
git clone https://github.com/ZJY-HSBL/SAGE-Seg.git
cd SAGE-Seg
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -e .
```

A CUDA-enabled PyTorch installation is recommended for full training. CPU execution is sufficient for unit tests and small-scale functional checks.

完整训练建议使用支持 CUDA 的 PyTorch 环境；单元测试和小规模功能验证可以直接在 CPU 上完成。

## Dataset layout · 数据集目录

### PASCAL VOC 2012

```text
<VOC_ROOT>/
├── JPEGImages/
├── SegmentationClass/
└── ImageSets/Segmentation/
    ├── train.txt
    └── val.txt
```

If your path is `VOCdevkit/VOC2012`, set `data.root` directly to the `VOC2012` directory.

如果数据位于 `VOCdevkit/VOC2012`，请将 `data.root` 直接设置为 `VOC2012` 目录。

### Cityscapes

```text
<CITYSCAPES_ROOT>/
├── leftImg8bit/
│   ├── train/
│   └── val/
└── gtFine/
    ├── train/
    └── val/
```

## Prepare semi-supervised splits · 生成半监督划分

VOC 1/8 example / VOC 1/8 示例：

```bash
python tools/prepare_splits.py \
  --dataset voc \
  --root /path/to/VOC2012 \
  --fraction 0.125 \
  --output-dir splits/voc \
  --seed 0
```

Cityscapes 1/8 example / Cityscapes 1/8 示例：

```bash
python tools/prepare_splits.py \
  --dataset cityscapes \
  --root /path/to/cityscapes \
  --fraction 0.125 \
  --output-dir splits/cityscapes \
  --seed 0
```

Update `data.labeled_list` and `data.unlabeled_list` in the corresponding YAML file after generating the split.

生成划分后，在对应 YAML 配置中更新 `data.labeled_list` 与 `data.unlabeled_list`。

## Training · 训练

```bash
python tools/train.py --config configs/voc.yaml --data-root /path/to/VOC2012
```

```bash
python tools/train.py --config configs/cityscapes.yaml --data-root /path/to/cityscapes
```

The default presets implement the following training logic:

默认配置实现以下训练逻辑：

```text
labeled image  ──weak──> student ──> supervised CE

unlabeled image ──weak A──> teacher ──> probability / pseudo-label
                └─weak B──> student ──> semantic consistency mask
                                      └─SRF──> aggregation mask M

teacher confidence = 1 - normalized entropy
z ~ Bernoulli(confidence)

z = 0: semantic mixing with labeled data
z = 1: strong augmentation

L = L_sup + delta * L_unsup
teacher <- EMA(student)
```

## Evaluation · 评估

```bash
python tools/evaluate.py \
  --config configs/voc.yaml \
  --checkpoint runs/SAGE-Seg-VOC/best.pt \
  --data-root /path/to/VOC2012
```

The evaluator reports mean IoU and per-class IoU.

评估脚本输出 mIoU 与各类别 IoU。

## Inference · 推理

```bash
python tools/infer.py \
  --config configs/voc.yaml \
  --checkpoint runs/SAGE-Seg-VOC/best.pt \
  --input demo.jpg \
  --output outputs/demo_mask.png
```

## Default experimental presets · 默认实验配置

| Setting | VOC 2012 | Cityscapes |
|---|---:|---:|
| Backbone | ResNet-101 | ResNet-101 |
| Decoder | DeepLabV3+ | DeepLabV3+ |
| Base LR | 0.001 | 0.005 |
| Head LR multiplier | 10× | 10× |
| Weight decay | 1e-5 | 1e-5 |
| Momentum | 0.9 | 0.9 |
| Total batch size | 16 | 16 |
| Labeled / unlabeled per step | 8 / 8 | 8 / 8 |
| Epochs | 80 | 190 |
| Crop | 512×512 | 512×1024 |
| Weak random scale | 0.5–2.0 | 0.5–2.0 |
| Horizontal flip | 0.5 | 0.5 |
| Consistency threshold ε | 0.7 | 0.7 |
| SRF area threshold τ | 16 | 32 |
| Unsupervised weight δ | 1.5 | 1.0 |

The area threshold and unsupervised weight are exposed in YAML because optimal values can vary with dataset, split protocol, and implementation details.

面积阈值与无监督损失权重均在 YAML 中开放配置，因为其最优值会随数据集、标注比例和实现细节变化。

## Tests · 测试

```bash
pytest -q
```

The test suite verifies semantic consistency masks, 4-connected sparse filtering, entropy confidence, Bernoulli routing interfaces, and semantic mixing behavior.

测试覆盖语义一致性掩码、四连通稀疏过滤、熵置信度、Bernoulli 路由接口以及语义混合逻辑。

## Notes · 说明

- No benchmark number is hard-coded into the training or evaluation pipeline.
- Reproducing exact benchmark results requires matching the dataset split, preprocessing, pretrained initialization, hardware, random seeds, and optimization details.
- The implementation keeps the algorithmic components modular so SCD, SRF, gating, or the segmentation backbone can be replaced independently.

- 训练与评估流程中没有硬编码任何性能结果。
- 若要严格对齐特定实验结果，需要保持数据划分、预处理、预训练初始化、硬件、随机种子与优化策略一致。
- SCD、SRF、门控机制与分割主干均采用模块化设计，可独立替换与扩展。
