# Learning-Note-Website-
In this learning note, I sort out the blogs, articles about mating learning/deep learning/computer vision on website.

## [A Year in Computer vision](www.themtank.org/pdfs/AyearofComputerVisionPDF.pdf)

### ConvNet 架构
近期，卷积神经网络在CV之外的领域也有很多新的应用，但总体说来，仍然主导者计算机视觉领域。下面列出2016年以来一些优秀的ConvNet架构。
- [Inception-v4,Inception-ResNet and the Impact of Residual Connections on Learning](http://arxiv.org/pdf/1602.07261v2.pdf)
- [Densely Connected Convolutional Networks](http://arxiv.org/pdf/1608.06993v3) 从ResNet的 identity/skip connection 获得灵感，通过在前馈中将每一层链接至其他层，将前面所有层的特征图作为当前层的输入。已有DenseNet的多个实现实例(keras，Tensorflow等):[https://github.com/liuzhuang13/DenseNet](https://github.com/liuzhuang13/DenseNet)
> DenseNets hasve several compelling advantages: they alleviate the vanishing-gradients problem, strengtheen feature propagation, encourage feature reuse, and substantially reduce the number of parameters.
- [FractalNet: Ultra-Deep Neural Networks without Residuals](https://arxiv.org/pdf/1605.07648v2.pdf) 使用不同深度（conv层的数量不同）的子路径，没有pass-trough或residual connection。

下面是一些补充的ConvNet架构：
- [Swapout: Learning an ensemble of deep architectures](https://arxiv.org/pdf/1605.06465v1.pdf) Swapout技术属于正则化的范畴，目的是防止特定层或者所有层中单元（unit）之间的共适应（co-adaptation）。该集成训练方法从多个架构中采样，包括**dropout**,**ResNet**和**随机生成**。
- [SqueezeNet](https://arxiv.org/pdf/1602.07360v4.pdf) SqueezeNet技术属于模型压缩减参的范畴。在与AlexNet达到相同准确率的情况下，SqueezeNet将参数数量和所需内存减少了510X。

以下几篇论文都处理了ConvNet中的旋转不变性（rotation-invariant）。每种方法都通过更有效的参数利用提高了旋转不变性，并且最终获得全局的旋转同变性（global rotation equivariance）。
- [Harmonic Networks: Deep Translation and Roatation Equivariance](https://arxiv.org/pdf/1612.04642v1.pdf)
- [Group Equivariant Convolutional Networks(G-CNNs)](https://arxiv.org/pdf/1602.07576v3.pdf)
- [Steerable CNNs](https://arxiv.org/pdf/1612.08498v1.pdf)

### Residual Network
微软的[ResNet](https://arxiv.org/pdf/1512.03385v1.pdf)获得成功，使得残差网络以及变体在2016年变得很流行。[Multi-ResNet](https://arxiv.org/pdf/1609.05672v3.pdf)在CIFAR-10和CIFAR-100上分别获得了3.65%和18.27%的2016年度最低误差率。



