# SAGE-Seg algorithm notes / SAGE-Seg 算法说明

SAGE-Seg uses a student network and an EMA teacher. The student is optimized with a supervised loss on labeled samples and an adaptive unsupervised loss on unlabeled samples.

SAGE-Seg 使用学生网络与 EMA 教师网络。学生模型在标记样本上计算监督损失，在未标记样本上计算自适应无监督损失。

## 1. Total objective / 总体目标

`L = L_sup + delta * L_unsup`

`L_sup` is pixel-wise cross entropy on weakly augmented labeled images.

`L_sup` 为弱增强标记图像上的像素级交叉熵。

## 2. SCD / 语义一致性判别

For teacher and student class-probability vectors at each pixel, compute cosine similarity. Pixels above `epsilon` are considered consistent. Pixels below the threshold are candidate replacement locations.

对每个像素位置的教师/学生类别概率向量计算余弦相似度。高于 `epsilon` 的像素视为一致，低于阈值的像素进入候选替换区域。

## 3. SRF / 稀疏区域过滤

The binary inconsistency map is decomposed with 4-connected components. Components whose area is smaller than `tau` are suppressed so isolated noisy pixels do not trigger semantic replacement.

二值不一致区域采用四连通分量分解。面积小于 `tau` 的分量被过滤，从而避免零散噪声像素触发语义替换。

## 4. Confidence-adaptive gate / 置信度自适应门控

Sample confidence is defined as one minus normalized predictive entropy. A Bernoulli variable is sampled from this confidence. High-confidence samples more often follow the strong-augmentation path; low-confidence samples more often follow the labeled-unlabeled semantic-mixing path.

样本置信度定义为 `1 - 归一化预测熵`。以该置信度为 Bernoulli 参数进行采样。高置信度样本更倾向于走强增强路径，低置信度样本更倾向于与标记数据进行语义混合。

## 5. Semantic mixing / 语义混合

`M=1` retains weakly augmented unlabeled content and its teacher pseudo-label. `M=0` inserts labeled image content and the corresponding ground-truth mask.

`M=1` 保留弱增强未标记图像及教师伪标签；`M=0` 替换为标记图像内容及其真实标签。

## 6. EMA update / EMA 更新

After each student optimization step:

`theta_teacher <- decay * theta_teacher + (1 - decay) * theta_student`

每次学生模型更新后，以指数移动平均方式更新教师参数。
