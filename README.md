# TwinLiteNet: An Efficient and Lightweight Model for Driveable Area and Lane Segmentation in Self-Driving Cars

## 模型结构修改:
1) 为了高效部署（rk3588），修改 ESPNet_Encoder 中的 PAM、CAM 结构，使用高效卷积替代低效算子；
2) 将分类损失函数从 Focal Loss 修改为 BCE；
3) 去掉一个 head 分支，变成单一前景类别的分割模型；


