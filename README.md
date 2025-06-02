# Deep Learning Project2 
本实验完成了两个任务。第一个任务是设计神经网络模型 ResNet18Plus，在 CIFAR-10 数据集上达到了最高 $0.9563$ 的正确率；第二个任务是完成与 BN 有关的可视化任务。

## 1. Task 1
首先从 https://drive.google.com/file/d/1N-48n3Ai5sBDV3wtHu4bvpDDcqNZJNSA/view?usp=sharing 下载实验结果和模型权重，其中包含两个文件夹 `reports` 和 `results` ，将其放在根目录下。之后在根目录下运行指令即可进行最优模型测试：
```
python task1.py
```
如果需要训练模型，则需要将 `task1.py` 中最后部分的注释转换为代码，重新使用上述指令运行即可。

如果需要查看运行结果，使用如下指令：
```
tensorboard --logdir=results
```
即可在浏览器中查看 task 1 涉及到的所有模型与训练策略的结果。

## 2. Task 2
实验结果已经存储在 `reports` 文件夹中，包含模型权重和结果可视化，在之前已经一并下载。如果想要复现实验，可以使用如下指令：
```
python task2.py
```
