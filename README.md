# Learning-Note-Website-
In this learning note, I sort out the blogs, articles about mating learning/deep learning/computer vision on website.

## [A Year in Computer vision](www.themtank.org/pdfs/AyearofComputerVisionPDF.pdf)
本文参考机器之心微信公众号文章 [“计算机视觉这一年”](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650733850&idx=1&sn=ee05c1e715621e41643cd6af5627a013&chksm=871b3964b06cb0728981e6500c700fa71272726c66b3fee1dfd23c5d18de0205873767bdf973&scene=38#wechat_redirect)


## PART Two
### Segmentation
**图像分割**指将整幅图片划分成不同的像素组（pixel group），每一组像素可以被标记和分类。**语义（semantic）分割** 能够进一步理解一幅图片中每个像素归属什么类别，如属于一只猫，一辆汽车还是其他类别。**实例（instance）分割**能够更近一步地将一幅图片中每个实例用不同颜色区分开来。图像分割是目前计算机视觉应用在自动驾驶领域中的关键。

以上三篇图像分割的文章均出自FARI，模型架构大同小异。可以参考Facebook Research的文章[Learning to Segment](https://research.fb.com/learning-to-segment/)

- DeepMask [Learning to Segment Object Candidates](https://arxiv.org/pdf/1506.06204v2.pdf)(NIPS 2015) 生成分割图（mask prediction）与物体预测分（object score prediction）两个任务同时进行，两个任务共享（share）模型的前几层;
> DeepMask employs a fairly traditional feedforward deep network design. In such networks, with progressively deeper network stages information is more abstract and semantically meaningful. For example, early layers in a deep net might capture edges and blobs, while upper layers tend to capture more semantic concepts such as the presence of an animal’s face or limbs. By design, these upper-layer features are computed at a fairly low spatial resolution (for both computational reasons and in order to be invariant to small shifts in pixel locations). This presents a problem for mask prediction: **The upper layer features can be used to predict masks that capture the general shape on an object but fail to precisely capture object boundaries**.([Learning to Segment](https://research.fb.com/learning-to-segment/), 2016)
- SharkMask [Learning to Refine Object Segments](https://arxiv.org/pdf/1603.08695v2.pdf)(*NIPS*，2015) 在DeepMask基础上进一步提出refinement module提高分割效果;
> SharpMask refines the output of DeepMask, generating higher-fidelity masks that more accurately delineate object boundaries. While DeepMask predicts coarse masks in a feedforward pass through the network, SharpMask reverses the flow of information in a deep network and refines the predictions made by DeepMask by using features from progressively earlier layers in the network. Think of it this way: To capture general object shape, you have to have a high-level understanding of what you are looking at (DeepMask), but to accurately place the boundaries you need to look back at lower-level features all the way down to the pixels (SharpMask). In essence, we aim to make use of information from all layers of a network, with minimal additional overhead.([Learning to Segment](https://research.fb.com/learning-to-segment/), 2016)
- Multi-Path Net [A MultiPath Network for Object Detection](https://arxiv.org/pdf/1604.02135v2.pdf)(*BMVC*，2016)
> DeepMask knows nothing about specific object types, so while it can delineate both a dog and a sheep, it can’t tell them apart. Plus, DeepMask is not very selective and can generate masks for image regions that are not especially interesting. So how do we narrow down the pool of relevant masks and identify the objects that are actually present? ([Learning to Segment](https://research.fb.com/learning-to-segment/), 2016)

以下两篇文章的模型相比于FAIR的DeepMask，SharpMask和MultiPath-Net有很大改变；对图像分割中scale和localisation处理得更好。

- [DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs](https://arxiv.org/pdf/1606.00915v1.pdf) 
- [Simple Does It: Weakly Supervised Instance and Semantic Segmentation](https://arxiv.org/pdf/1603.07485v2.pdf) 弱监督情况下（训练集中不需要原始图片对应的精准分割结果，而仅仅需要图像中每个instance的Bounding Box）实现图像语义标准和实例分割。

以下文章的模型都是基于[Fully Convolutional Networks for Semantic Segmentation](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)(*CVPR*，2015)提出的全卷积网络（FCN），进一步利用端到端网络共享有用的信息，无须处理分类这一子任务。

- [The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation](https://arxiv.org/pdf/1611.09326v2.pdf) 文章拓展[DenseNet](https://arxiv.org/pdf/1608.06993.pdf)(*CVPR*，2017)到FCN，同时抑制了特征层过多的问题；同时文章提出的基于dense block的上采样路径（upsampling path）比其他传统的上采样路径表现更好。
- [Fully Convolutional Instance-aware Semantic Segmentation](https://arxiv.org/pdf/1611.07709v1.pdf) 文章提出了首个端到端全连接的instance-aware 语义分割方法，基于paper[Instance-sensitive fully convolutional networks](https://arxiv.org/pdf/1603.08678.pdf)(ECCV, 2016)拓展而来。

以下文章2016年在**healthcare application**领域的分割模型：
- [U-net:Convolutional networks for biomedical image segmentation](https://arxiv.org/pdf/1505.04597.pdf)(*MICAI*，2015)，本文提出的U-Net结构在众多Image-to-image任务中应用非常广泛。

关于医疗影像分割的文章，后续有待补充...

### Super-res/Colourisation/Style Transfer

#### Style Transfer
2015年两款照片风格化的应用被广泛关注：[Prisma](https://prisma-ai.com/)和[Artomatix](https://services.artomatix.com/) 风格迁移本身是一个很早就提出来的技术，自从2015年一篇[A Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1508.06576v2.pdf)发表使它开始跟神经网络结合起来，核心在于Gram矩阵的提出，可以有效的提取一幅图片的风格特征。
陆续有研究者在其基础上进行expend：
- [Exploring the Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1602.07188v2.pdf)
- [Artistic style transfer for videos](https://arxiv.org/pdf/1604.08610v2.pdf) 将风格迁移应用到视频。
- [A Learned Representation for Artistic Style](https://arxiv.org/pdf/1610.07629v5.pdf)(*ICLR*，2017) 不再像以往将一幅图像的内容和一幅艺术画的风格结合，谷歌的这篇文章将迁移网络（transfer net）进行了简单的修改，*conditional instance normalization*，可以将一幅图片和多个艺术画的内容结合，从A风格渐变至B风格。参考Research Blog[Supercharging Style Transfer](https://research.googleblog.com/2016/10/supercharging-style-transfer.html)。

#### Colourisation
图像的着色一般指将黑白（monochrome）图片转化为全色（full-colour）图片的过程。这一过程传统上是由人们手工完成的，在2016年，这一着色过程可以借由深度学习自动完成。图像的着色过程非常有趣，因为网络需要基于自身习得的对图像中物体位置、纹理和环境的理解为每个区域分配最可能的颜色。2016年三个最具影响力的工作：
- [Colorful Image Colorization](https://arxiv.org/pdf/1603.08511v5.pdf) 文章得到的着色效果在图灵测试中可以骗过32%的测试者。
- [Learning Representations for Automatic Colorization](https://arxiv.org/pdf/1603.06668v2.pdf)
- [Let there be Color!](http://hi.cs.waseda.ac.jp/~iizuka/projects/colorization/en/) 文章创新点在于端到端的学习全局和局部的图像先验，在图灵测试中欺骗92.6%的测试者，baseline是70%，ground-truth是97.7%。

#### Action Recognition
关于动作识别的文章，后续有待补充...



## PART Threee
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




