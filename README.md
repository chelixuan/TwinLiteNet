# TwinLiteNet: An Efficient and Lightweight Model for Driveable Area and Lane Segmentation in Self-Driving Cars

## 模型结构修改:
1) 为了高效部署（rk3588），修改 ESPNet_Encoder 中的 PAM、CAM 结构，使用高效卷积替代低效算子；
2) 将分类损失函数从 Focal Loss 修改为 BCE；
3) 去掉一个 head 分支，变成单一前景类别的分割模型
4）修改 dataloader，只能将所有训练数据放到同一个文件夹下的问题，通过 find_images 函数，只要放在同一根目录下即可；
   同时增加 train、val 参数，指定训练集、验证集路径，无需每次更换数据集都需要去 dataloader 中修改路径；
5）增加 validation 可选参数，决定是否每个训练 epoch 都要 validation（有时候无验证集）；

