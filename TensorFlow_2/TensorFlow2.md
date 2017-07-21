---
title: TensorFlow学习笔记
date: 2017-07-17
tags: TensorFlow
---

# TensorFlow学习笔记（2）

> 借助 ***迁移学习（Transfer Learning）*** 知识，使用**Inception v3**模型对我们已有的样本图片进行再训练，得到指定图片样本集下的分类模型

## 1.简单介绍

​	该笔记将介绍如果借助TensorFlow中给出的例子，很快解决我们实际工作中亟待解决的图像识别或者分类问题（Image Recognition）。其实TensorFlow官方文档已经给出了非常详细的说明（参考资料2，3），只是缺少了使用再次训练的模型（Retrained Model），进行分类测试和对模型评估的操作过程（官方文档有说明，但是针对第一次接触TensorFlow的小白用户，还是不够详细）。首次接触需要了解的背景知识：

- [ImageNet](http://www.image-net.org/)：Inception-v3模型依赖的样本集合
- "Model"前辈：[QuocNet(2012)](http://static.googleusercontent.com/media/research.google.com/en//archive/unsupervised_icml2012.pdf)，[AlexNet](http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf)，[Inception(GoogLeNet)(2014)](https://arxiv.org/abs/1409.4842)，[BN-Inception-v2](https://arxiv.org/abs/1502.03167)
- [Inception-v3](https://arxiv.org/abs/1512.00567)：具体和前辈们有什么差别，......，Google讲的比我好
- [Inception-v4](https://arxiv.org/abs/1602.07261)：v3还没有研究清楚，v4又于2016年2月横空出世

## 2.官方实例Training on Flowers

- TensorFlow: ***商业*** ***MacOS or Linux*** ***Google Brain***
  ***Google research and production needs***
  ***前任：DistBelif*** 
- PyTorch: ***商业*** ***Facebook*** ***Lua-based Torch Framework*** ***Pythonic***
- Theano: ***学术*** ***Any OS***  ***Python库*** ***与NumPy紧密集成***
- Keras: ***基于Theano和TensorFlow的深度学习库***

[针对Google的TenserFlow和Facebook的PyTorch的比较](https://medium.com/@dubovikov.kirill/pytorch-vs-tensorflow-spotting-the-difference-25c75777377b)：

***TensorFlow***: 

- 更适合生产环境的模型开发


- 开发需要在手机平台上部署的模型
- 丰富的社区支持和更为全面的文档说明
- 丰富的学习资源, 如[MOOC](https://www.udacity.com/course/deep-learning--ud730)
- 可视化工具Tensorboard
- 大规模分布式的模型训练

***PyTouch***

- 相对更友好的开发和调试环境
- *Love all things Pythonic*

[对比深度学习十大框架：TensorFlow最流行但并不是最好](https://zhuanlan.zhihu.com/p/24687814)

## 3.TensorFlow环境搭建

**操作系统**：MacOS

**依赖工具**：Virtualenv，Python3.6

**参考链接**：

- [Installing TensorFlow on Mac OS X](https://www.tensorflow.org/install/install_mac)
- [配合Vagrant在Mac上搭建一个干净的TensorFlow环境](https://juejin.im/post/58a85f7975c4cd340fa497bd)

***TensorFlow在Mac上编译安装方法*** [参考链接](https://www.tensorflow.org/install/install_sources)

虽然Mac下使用 ```pip install tensorflow``` 就可以安装tensorflow，但没有CPU和GPU加速，而且会出现一堆如下警告:

```python
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE3 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
```

所以决定自己编译源码安装（记住一句话：”没有强迫症的程序员不是好运维“），无GPU加速。

1. clone源码

   ***git clone https://github.com/tensorflow/tensorflow***

2. 安装依赖

   - bazel ***brew install bazel***
   - TensorFlow Python依赖: ***six*** ***numpy*** ***wheel***

3. 配置安装

   按下面命令配置源码，我的mac pro没有NVIDA的显卡，不需要配置GPU加速，所以一路回车就可以。配置后会自动下载依赖的gz，由于大家都懂的网络原因，会出现各种Timeout，不要怕，上VPN就能解决问题，这个下载过程很慢，耐心等待。

   ***./configure***

4. 编译源码

   这是关键步骤，因为mac没有GPU，只能优化CPU，采用-march=native参数会根据本机CPU特性进行编译（mac pro支持SSE4.2）。这个过程也会下载各种依赖，编译过程大概需要1-2小时，耐心等待。

   ***bazel build --config=opt --copt=-march=native //tensorflow/tools/pip_package:build_pip_package***

5. 生成whl包并安装

   ***bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg***
   ***pip install /tmp/tensorflow_pkg/tensorflow-1.2.1-py2-none-any.whl***

6. TensorFlow测试

   ```python
   # Python
   import tensorflow as tf
   hello = tf.constant('Hello, TensorFlow!')
   sess = tf.Session()
   print(sess.run(hello))
   ```

## 4. 参考资料

- 1. ***迁移学习（Transfer Learning）***：[DeCAF: A Deep Convolutional Activation Feature for Generic Visual Recognition](https://arxiv.org/pdf/1310.1531v1.pdf)
  2. [TensorFlow Tutorials--Image Recognition](https://www.tensorflow.org/tutorials/image_recognition)
  3. [How to Retrain Inception's Final Layer for New Categories](https://www.tensorflow.org/tutorials/image_retraining)