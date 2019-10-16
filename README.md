# fast.ai v3 part 2 PyTorch learning group
This is the repository for the upcoming learning group meetup in October based on [fast.ai v3 part 2 course](https://course.fast.ai/part2), [fastai v2 library development](https://github.com/fastai/fastai_dev), and [PyTorch v1.2](https://pytorch.org) course taking place in Vienna.
(See also the repository from the previous [fastai pytorch course in Vienna v1](https://github.com/MicPie/fastai-pytorch-course-vienna) based on the [fast.ai v3 part 1 course material](https://course.fast.ai).)

**[❗ Please register in order to get the updates for the meetups.](https://docs.google.com/forms/d/e/1FAIpQLScCEnJfFcyLQvT0rGd6HoN4oZf1lAe4ZnfWH1dfnXIQFyAMfQ/viewform)**


## ❔ Prerequisites
*For this learning group meetup you are expected to have basic knowledge of deep learning or have gone through the [fast.ai v3 part 1 course material](https://course.fast.ai) and to have at least one year experience with programming. You should feel comfortable with programming in Python as well as having basic knowledge in Calculus and Linear Algebra. Some machine learning background is advised to make best use of the course.*


## 📅 Dates
#### 🐍 PyTorch Python part
* Lesson 8: 16.10.2019 18:00-20:00 - Matrix multiplicatio; Forward and backward passes - Michael Pieler
* Lesson 9: 6.11.2019 18:30-20:30 - Loss functions, optimizers, and the training loop - Liad Magen & Thomas Keil
* Lesson 10: 20.11.2019 18:30-20:30 - Looking inside the model - Albert Rechberger, Moritz Reinhardt, Johannes Hofmanninger
* Lesson 11: exact day to be announced soon
* Lesson 12: 18.12.29019 18:30-20:30
#### 🎄 Xmas break
#### 🧮 Swift4Tensorflow part
* Lesson 13: tba
* Lesson 14: tba

*Note: All the learning group meetups will take place at Nic.at, Karlsplatz 1, 1010 Wien.*


## 📖 Lesson material
### Lesson 8 - Matrix multiplication; forward and backward passes
(The first lesson already starts with number 8, because the part 1 course contained 7 lessons.)
* **To dos before the lesson:**
  * **watch the [fastai lesson 8](https://course.fast.ai/videos/?lesson=8) ([timlee lesson notes](https://forums.fast.ai/t/lesson-8-notes/41442/22))**
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
  * **watch the [fastai lesson 9](https://course.fast.ai/videos/?lesson=9)**
  * **run the [lesson 9 notebook]()**
* tba

### Lesson 10 - Looking inside the model
* **To dos before the lesson:**
  * **watch the [fastai lesson 10](https://course.fast.ai/videos/?lesson=10)**
  * **run the [lesson 10 notebook]()**
* tba 

### Lesson 11 - Data Block API, and generic optimizer
* **To dos before the lesson:**
  * **watch the [fastai lesson 11](https://course.fast.ai/videos/?lesson=11)**
  * **run the [lesson 11 notebook]()**
* tba 

### Lesson 12 - Advanced training techniques; ULMFiT from scratch
* **To dos before the lesson:**
  * **watch the [fastai lesson 12](https://course.fast.ai/videos/?lesson=12)**
  * **run the [lesson 12 notebook]()**
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
