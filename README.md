# fast.ai v3 part 2 PyTorch learning group
This is the repository for the upcoming learning group meetup in October based on [fast.ai v3 part 2 course](https://course.fast.ai/part2), [fastai v2 library development](https://github.com/fastai/fastai_dev), and [PyTorch v1.2](https://pytorch.org) course taking place in Vienna.
(See also the repository from the previous [fastai pytorch course in Vienna v1](https://github.com/MicPie/fastai-pytorch-course-vienna) based on the [fast.ai v3 part 1 course material](https://course.fast.ai).)

**[❗ Please register in order to get the updates for the meetups.](https://docs.google.com/forms/d/e/1FAIpQLScCEnJfFcyLQvT0rGd6HoN4oZf1lAe4ZnfWH1dfnXIQFyAMfQ/viewform)**


## ❔ Prerequisites
*For this learning group meetup you are expected to have basic knowledge of deep learning or have gone through the [fast.ai v3 part 1 course material](https://course.fast.ai) and to have at least one year experience with programming. You should feel comfortable with programming in Python as well as having basic knowledge in Calculus and Linear Algebra. Some machine learning background is advised to make best use of the course.*


## 📅 Dates
#### 🐍 PyTorch Python part
* Lesson 8: 16.10.2019 18:00-20:00 - Matrix multiplication; Forward and backward passes - Michael Pieler
* Lesson 9: 6.11.2019 18:30-20:30 - Loss functions, optimizers, and the training loop - Liad Magen & Thomas Keil
* Lesson 10: 20.11.2019 18:30-20:30 - Looking inside the model - Albert Rechberger, Moritz Reinhardt, Johannes Hofmanninger
* Lesson 11: 10.12.2019 18:30-20:30 - Data Block API, and generic optimizer - Sebastian Dürr
* Lesson 12: 18.12.29019 18:30-20:30 - Advanced training techniques; ULMFiT from scratch - ?
#### 🎄 Xmas break
#### 🧮 Swift4Tensorflow part
* Lesson 13: 15.01.2020 18:30-20:30 - ?
* Lesson 14: 05.02.2020 18:30-20:30 - ?

*Note: All the learning group meetups will take place at Nic.at, Karlsplatz 1, 1010 Wien.*


## 📖 Lesson material
### Lesson 8 - Matrix multiplication; forward and backward passes
(The first lesson already starts with number 8, because the part 1 course contained 7 lessons.)
* **To dos before the lesson:**
  * **watch the [fastai lesson 8](https://course.fast.ai/videos/?lesson=8) ([lesson notes](https://forums.fast.ai/t/lesson-8-notes/41442/22))**
  * **run the [matrix multiplication](https://github.com/fastai/course-v3/blob/master/nbs/dl2/01_matmul.ipynb) and the [forward and backward pass](https://github.com/fastai/course-v3/blob/master/nbs/dl2/02_fully_connected.ipynb) notebooks**
  * *Do not worry, the first lesson is quite dense and we will tackle the building blocks piece by piece! :-)*
  * [Matrix multiplication on German Wikipedia](https://de.wikipedia.org/wiki/Matrizenmultiplikation) (the German version has  better visualisations)
  * [Animated matrix multiplication](http://matrixmultiplication.xyz)
  * [Broadcasting visualisation](https://jakevdp.github.io/PythonDataScienceHandbook/02.05-computation-on-arrays-broadcasting.html)
  * Refresh your PyTorch basics with the [lerning material from our previous fast.ai v3 part 1 learning group](https://github.com/MicPie/fastai-pytorch-course-vienna#lesson-1---intro-to-fastai-and-pytorch).
  * Get familiar with [PyTorch einsum](https://rockt.github.io/2018/04/30/einsum) to get more intuition for matrix multiplication.
  * [What is torch.nn *really*?](https://pytorch.org/tutorials/beginner/nn_tutorial.html) (This nicely explains the steps needed for training a deep learning model with PyTorch. It covers torch.nn, torch.optim, Dataset, and DataLoader. This setup is a "blueprint" for a deep learning library based on PyTorch.)
* PyTorch basics: [introduction](https://github.com/MicPie/fastai-pytorch-course-vienna/blob/master/PyTorch_1_Intro.ipynb), [torch.nn](https://github.com/MicPie/fastai-pytorch-course-vienna/blob/master/PyTorch_2_torchnn.ipynb), [view vs. permute](https://github.com/MicPie/pytorch/blob/master/view_and_permute.ipynb), [debugging](https://github.com/MicPie/fastai-pytorch-course-vienna/blob/master/PyTorch_3_debugging.ipynb), and [scaled dot product attention as a matrix multiplication example](https://github.com/MicPie/pytorch/blob/master/attention.ipynb)
* [fastai v2 dev test setup](https://github.com/fastai/fastai_dev/blob/master/dev/00_test.ipynb)
* Go deeper with DL [debugging](https://github.com/MicPie/fastai-pytorch-course-vienna/blob/master/PyTorch_3_debugging.ipynb), [troubleshooting (pdf](https://fullstackdeeplearning.com/assets/slides/fsdl_10_troubleshooting.pdf) or [video)](https://www.youtube.com/watch?v=GwGTwPcG0YM&feature=youtu.be), and [how to avoid it in the first place (i.e., the Karpathy recipe)](http://karpathy.github.io/2019/04/25/recipe/).
* [Why understanding backprop can be important for debugging.](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b)
* [Xavier Glorot and Kaiming He init](https://pouannes.github.io/blog/initialization/)
* Publications:
  * [Matrix calculus for DL (web)](https://explained.ai/matrix-calculus/index.html) [(arxiv)](https://arxiv.org/abs/1802.01528)
  * [Xavier Glorot init](http://proceedings.mlr.press/v9/glorot10a.html)
  * [Kaiming He init](https://arxiv.org/abs/1502.01852)
  * [Fixup init](https://arxiv.org/abs/1901.09321)
  * [Batch norm](https://arxiv.org/pdf/1502.03167) and [how does it help optimization](https://arxiv.org/pdf/1805.11604)
  * If you want to present one of the papers in this or the next lectures reach out to us via email! :-)
* If you want to know more about [matrix multiplication & Co. on your (Nvidia) GPU](https://devblogs.nvidia.com/cutlass-linear-algebra-cuda/) (or why everything should be a multiple of 8 for super fast calculations on Nvidia GPUs).
* PyTorch code examples: [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
* [Visual information theory](https://colah.github.io/posts/2015-09-Visual-Information/), [KL & cross entropy (see section 1., question 3.)](https://www.depthfirstlearning.com/2018/InfoGAN)
* [Mish](https://forums.fast.ai/t/meet-mish-new-activation-function-possible-successor-to-relu/53299) (new activation function)
* Project ideas:
  * ReLU with different backward pass function
  * ? (we mentioned something else, but I forgot it, if you know it, please make a pull request)

### Lesson 9 - Loss functions, optimizers, and the training loop
* **To dos before the lesson:**
  * **watch the [fastai lesson 9](https://course.fast.ai/videos/?lesson=9) ([lesson notes](https://medium.com/@lankinen/fast-ai-lesson-9-notes-part-2-v3-ca046a1a62ef))**
  * **run the lesson 9 notebooks: [why sqrt(5)](https://github.com/fastai/course-v3/blob/master/nbs/dl2/02a_why_sqrt5.ipynb), [init](https://github.com/fastai/course-v3/blob/master/nbs/dl2/02b_initializing.ipynb), [minibatch training](https://github.com/fastai/course-v3/blob/master/nbs/dl2/03_minibatch_training.ipynb), [callbacks](https://github.com/fastai/course-v3/blob/master/nbs/dl2/04_callbacks.ipynb), and [anneal](https://github.com/fastai/course-v3/blob/master/nbs/dl2/05_anneal.ipynb)**
* [Lesson presentation slides](https://github.com/MicPie/fastai-pytorch-course-vienna-v2/blob/master/fastai%20part%20II%20-%20lesson%209.pdf)
* [A super intro to NN weight initialization from cs231n](https://cs231n.github.io/neural-networks-2/#init)
* [Weights initialization](https://madaan.github.io/init/) - blog post about Xavier Initialization
* [Neural Network visualizer playground](https://playground.tensorflow.org/) - allows you to play with parameters such as learning rate, batch size and regularization, and see the result while training directly on the browser
* [What is torch.nn?](https://pytorch.org/tutorials/beginner/nn_tutorial.html)
* [Common neural network mistakes](https://twitter.com/karpathy/status/1013244313327681536) (Twitter thread, combine with publication below.)
* [Pytorch under the hood](https://speakerdeck.com/perone/pytorch-under-the-hood)
* [Bias in NN?](https://www.quora.com/What-is-bias-in-artificial-neural-network)
* [Correlation and dependence](https://en.wikipedia.org/wiki/Correlation_and_dependence) (have a look a the correlation coefficient figure)
* Floating point basics: We (usually) use [FP32](https://en.wikipedia.org/wiki/Single-precision_floating-point_format#IEEE_754_single-precision_binary_floating-point_format:_binary32) or [FP16](https://en.wikipedia.org/wiki/Half-precision_floating-point_format#IEEE_754_half-precision_binary_floating-point_format:_binary16) (in combination with FP32) for [mixed precision training](https://forums.fast.ai/t/mixed-precision-training/20720) ([detailed information on floating point arithmetic](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)).
* [Efficient Methods and Hardware for Deep Learning](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture15.pdf) (quantization, ternary net, & Co.)
* Publications:
  * [Taxonomy of Real Faults in Deep Learning Systems](https://arxiv.org/abs/1910.11015) (see page 7 for a nice overview)

### Lesson 10 - Looking inside the model
* **To dos before the lesson:**
  * **watch the [fastai lesson 10](https://course.fast.ai/videos/?lesson=10) ([lesson notes](https://medium.com/@lankinen/fast-ai-lesson-10-notes-part-2-v3-aa733216b70d))**
  * **run the lesson 10 notebooks: [foundations](https://github.com/fastai/course-v3/blob/master/nbs/dl2/05a_foundations.ipynb), [early stopping](https://github.com/fastai/course-v3/blob/master/nbs/dl2/05b_early_stopping.ipynb), [CUDA CNN hooks init](https://github.com/fastai/course-v3/blob/master/nbs/dl2/06_cuda_cnn_hooks_init.ipynb), [batch norm](https://github.com/fastai/course-v3/blob/master/nbs/dl2/07_batchnorm.ipynb)**
* [Python data model for \_\_dunder\_\_ & Co.](https://docs.python.org/3/reference/datamodel.html), [a Guide to Python's Magic Methods](https://rszalski.github.io/magicmethods), and [exceptions](https://docs.python.org/3/library/exceptions.html)
* [Illustrated Explanation of Performing 2D Convolutions Using Matrix Multiplications](https://medium.com/@_init_/an-illustrated-explanation-of-performing-2d-convolutions-using-matrix-multiplications-1e8de8cd2544)
* [An infinitely customizable training loop with Sylvain Gugger](https://www.youtube.com/watch?v=roc-dOSeehM)
* Publications:
  * [Batch norm](https://arxiv.org/abs/1502.03167)
  * [Layer norm](https://arxiv.org/abs/1607.06450)
  * [Instance norm](https://arxiv.org/abs/1607.08022)
  * [Group norm](https://arxiv.org/abs/1803.08494)
  * [Spectral norm](https://arxiv.org/abs/1802.05957)
  * [Revisiting Small Batch Training for Deep Neural Networks](https://arxiv.org/abs/1804.07612)
* Debugging: "Debugging is twice as hard as writing the code in the first place. Therefore, if you write the code as cleverly as possible, you are, by definition, not smart enough to debug it." - Kernighan’s law

### Lesson 11 - Data Block API, and generic optimizer
* **To dos before the lesson:**
  * **watch the [fastai lesson 11](https://course.fast.ai/videos/?lesson=11) ([lesson notes](https://medium.com/@lankinen/fast-ai-lesson-11-notes-part-2-v3-6d28e17509f4))**
  * **run the lesson 11 notebooks: [LSUV](https://github.com/fastai/course-v3/blob/master/nbs/dl2/07a_lsuv.ipynb), [Data Block API](https://github.com/fastai/course-v3/blob/master/nbs/dl2/08_data_block.ipynb), [Optimizers](https://github.com/fastai/course-v3/blob/master/nbs/dl2/09_optimizers.ipynb), [new Learner](https://github.com/fastai/course-v3/blob/master/nbs/dl2/09b_learner.ipynb), [progress bar for Learner](https://github.com/fastai/course-v3/blob/master/nbs/dl2/09c_add_progress_bar.ipynb), and [Data Augmentation](https://github.com/fastai/course-v3/blob/master/nbs/dl2/10_augmentation.ipynb)**
* [LSUV notebook](https://gist.github.com/simongrest/52404966f0c46f750a823a44618bb06c) (see also publication below)
* [CNNs: from the Basics to Recent Advances (Dmytro Mishkin)](http://ai.ms.mff.cuni.cz/~sui/mishkin.pdf) (see also publication below)
* [Imagenette dataset](https://github.com/fastai/imagenette)
* [Exponentially weighted averages, momentum, RMSprob, and Adam](https://forums.fast.ai/t/lesson-11-discussion-and-wiki/43406/336)
* [Optimizer setup in fastai v2](https://github.com/fastai/fastai_dev/blob/master/dev/12_optimizer.ipynb)
* Publications:
  * [All you need is a good init (LSUV)](https://arxiv.org/abs/1511.06422)
  * [Systematic evaluation of CNN advances on the ImageNet](https://arxiv.org/pdf/1606.02228)
  * [L2 Regularization versus Batch and Weight Normalization](https://arxiv.org/abs/1706.05350)
  * [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) (Adam with decoupled weight decay, i.e., AdamW)
  * Blog post on [understanding the LARS/LAMB optimizer](https://towardsdatascience.com/an-intuitive-understanding-of-the-lamb-optimizer-46f8c0ae4866)
* ***Xmas project(s): Please suggest a project, dataset, kaggle competitions, etc.!***
  * [APTOS2019 blindness-detection](https://www.kaggle.com/c/aptos2019-blindness-detection) (in combination with the interesting starting points outlined in a [blog post about transfer learning for medical imaging by Google](https://ai.googleblog.com/2019/12/understanding-transfer-learning-for.html))
  * [Understanding clouds](https://www.kaggle.com/c/understanding_cloud_organization)

### Lesson 12 - Advanced training techniques; ULMFiT from scratch
* **To dos before the lesson:**
  * **watch the [fastai lesson 12](https://course.fast.ai/videos/?lesson=12) ([lesson notes](https://medium.com/@lankinen/fast-ai-lesson-12-notes-part-2-v3-dd53bec89c0b))**
  * **run the lesson 12 notebooks: [MixUp & Label Smoothing](https://github.com/fastai/course-v3/blob/master/nbs/dl2/10b_mixup_label_smoothing.ipynb), [Mixed Precision Training](https://github.com/fastai/course-v3/blob/master/nbs/dl2/10c_fp16.ipynb), [Train Imagenette](https://github.com/fastai/course-v3/blob/master/nbs/dl2/11_train_imagenette.ipynb), [Transfer Learning](https://github.com/fastai/course-v3/blob/master/nbs/dl2/11a_transfer_learning.ipynb), [Text](https://github.com/fastai/course-v3/blob/master/nbs/dl2/12_text.ipynb), [AWD-LSTM](https://github.com/fastai/course-v3/blob/master/nbs/dl2/12a_awd_lstm.ipynb), [Language Model Pretraining](https://github.com/fastai/course-v3/blob/master/nbs/dl2/12b_lm_pretrain.ipynb), and [ULMFiT](https://github.com/fastai/course-v3/blob/master/nbs/dl2/12c_ulmfit.ipynb)**
* [MixUp, CutMix, and others for fastai](https://github.com/oguiza/fastai_extensions)
* [AugMix](https://github.com/google-research/augmix)
* [Albumentations image data augmentation library](https://albumentations.readthedocs.io/en/latest/)
* Publications:
  * [Bag of Tricks for Image Classification with CNNs](https://arxiv.org/abs/1812.01187) (Highly recommended!)
  * [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)
  * []()
  * []()
* tba 

### Lesson 13 - Basics of Swift for Deep Learning
* **To dos before the lesson:**
  * **watch the [fastai lesson 13](https://course.fast.ai/videos/?lesson=13)**
  * **run the [lesson 13 notebook]()**
* tba 

### Lesson 14 - C interop; Protocols; Putting it all together
* **To dos before the lesson:**
  * **watch the [fastai lesson 14](https://course.fast.ai/videos/?lesson=14)**
  * **run the [lesson 14 notebook]()**
* tba 


## 🗄️ Information material
### 📚 Course
* [fast.ai v3 part 2 course details](https://www.fast.ai/2019/06/28/course-p2v3/)
* [fast.ai v3 part 2 course material](https://course.fast.ai/part2) (this should be your first address if you are searching for something)
* [fast.ai v3 part 2 course notebooks](https://github.com/fastai/course-v3/tree/master/nbs/dl2)
* [fastai v1 docs](https://docs.fast.ai) (this should be your second address if you are searching for something)
* [fastai v2 dev repo](https://github.com/fastai/fastai_dev) (We will have a look at the notebooks used for the development of fastai v2 to see how the different parts end up in the library.)
* [fast.ai forum](https://forums.fast.ai) (this should be your third address if you are searching for something)
  * [fast.ai v3 part 2 sub-forum](https://forums.fast.ai/c/part2-v3)
  * [fastai v2 sub-forum](https://forums.fast.ai/c/fastai-users/fastai-v2)
  * [Study group in Austria thread on the fast.ai forum](https://forums.fast.ai/t/study-group-in-austria/26119/10)
* [TWiML fast.ai v3 part 2 study group material](https://github.com/jcatanza/Fastai-Deep-Learning-From-the-Foundations-TWiML-Study-Group)

### 🛠️ Preparation
 * [Preparation part from our first learning group](https://github.com/MicPie/fastai-pytorch-course-vienna/blob/master/README.md#preparation)
 * [General PyTorch Deep Learning ressources](https://github.com/MicPie/fastai-pytorch-course-vienna/blob/master/README.md#general-pytorch-deep-learning-ressources)

### 💡 Others
 * [Learning tips](https://github.com/MicPie/fastai-pytorch-course-vienna/blob/master/README.md#learning-tips)
 * [Do not forget, the path to mastery is not a straight line!](https://pbs.twimg.com/media/CX0hrijUAAABGIA.jpg:large) [(From the book "Chop Wood Carry Water".)](https://www.amazon.com/dp/153698440X)
 * Please feel free to send us suggestions!
