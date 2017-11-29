### GAN + Saliency Prediction
- [SalGAN: Visual Saliency Prediction with Generative Adversarial Networks](https://arxiv.org/pdf/1701.01081.pdf) 文章无论从网络架构还是目标函数的设计上都基本延续[Pix2Pix](http://arxiv.org/pdf/1611.07004.pdf)，不同点在于去除了skip connection，这一点直观上可以理解，因为目标图并不需要保留原图中的高频边缘信息，另外SalGAN中*generator*网络的encoder部分复用预训练好的VGG的前几层，在显著性预测训练过程中，只更新encoder后两层的参数；decoder部分的参数则随机初始化。另一方面值得关注的是对于这类问题的各式各样的量化指标。实现代码：https://imatge-upc.github.io/saliency-salgan-2017/
- [Supervised Adversarial Networks for Image Saliency Detection](https://arxiv.org/pdf/1704.07242.pdf) 很easy的工作，没有参考价值。

### GAN + Semantic Segmentation
- [SegAN: Adversarial Network with Multi-scale L1 Loss for Medical Image Segmentation](https://arxiv.org/pdf/1706.01805.pdf)
- [Semantic Segmentation using Adversarial Networks](https://arxiv.org/pdf/1611.08408.pdf)(*NIPS Workshop*，2016) Chainer实现：https://github.com/oyam/Semantic-Segmentation-using-Adversarial-Networks




### 眼动（Gaze）预测
- [Supervising Neural Attention Models for Video Captioning by Human Gaze Data](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yu_Supervising_Neural_Attention_CVPR_2017_paper.pdf)(*CVPR*，2017)


