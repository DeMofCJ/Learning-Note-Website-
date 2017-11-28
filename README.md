# Learning-Note-Website-
In this learning note, I sort out the blogs, articles about mating learning/deep learning/computer vision on website.

## [A Year in Computer vision](www.themtank.org/pdfs/AyearofComputerVisionPDF.pdf)
本文参考机器之心微信公众号文章[“计算机视觉这一年”]（https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650733850&idx=1&sn=ee05c1e715621e41643cd6af5627a013&chksm=871b3964b06cb0728981e6500c700fa71272726c66b3fee1dfd23c5d18de0205873767bdf973&scene=38#wechat_redirect）

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
ResNet的提出重点解决了当网络越深，梯度容易消失（vanishing）的问题，基于更深网络的ResNet获得比普通网络更好的性能，使人们更加相信"increased depth produces superior abstraction"这一理念。但ResNet之所以work也有另一种解释，即ResNet本质上是浅层网络的集成。跳跃连接（skip connection）的引入在某种程度上抵消了DNN的分层性质，允许更容易的反向传播，缓解了DNN中梯度消失（vanishing）和梯度爆炸（exploding）的问题。

### 残差学习、理论和改进
- [Wide Residual Networks](https://arxiv.org/pdf/1605.07146v3.pdf) 目前非常普遍的ResNet方法，通过增加网络的宽度和减少深度提升了网络的性能，缓解了逐渐减少的特征重用（diminishing feature reuse）的问题。
- [Deep Networks with Stochastic Depth](https://arxiv.org/pdf/1603.09382v3.pdf) 将dropout技术应用于整个层的神经元，而不是单个神经元，即"We start with very deep networks but during training, for each mini-batch, randomly drop a subset of layers and bypass them with the identity function"。优势：更快的训练，更高的准确率。
- [Learning Identity Mappings with Residual Gates](https://arxiv.org/pdf/1611.01260v2.pdf) "by using a scalar parameter to control each gate, we provide a way to learn identity mappings by optimizing only one parameter"
- [Residual Networks Behave Like Ensembles of Relatively Shallow Networks](https://arxiv.org/pdf/1605.06431v2.pdf) ResNet可以看成很多路径的集成。参考Quora：[https://www.quora.com/What-is-an-intuitive-explanation-of-Deep-Residual-Networks](https://www.quora.com/What-is-an-intuitive-explanation-of-Deep-Residual-Networks)
- [Identity Mappings in Deep Residual Networks](https://arxiv.org/pdf/1603.05027v3.pdf)
- [Multi-Residual Networks: Improving the Speed and Accuracy of Residual Networks](https://arxiv.org/pdf/1609.05672v3.pdf) "The proposed multi-residual network increases the number of residual functions in the residual blocks"，提倡ResNet集成的思想，支持ResNet架构变得更宽、更深。




