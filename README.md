# Influencer Centrality

The algorithm in this package is inspired about the paper **"Detecting Topic Authoritative Social Media Users: a
Multilayer Network Approach"**.
In this paper, the authors propose a method capable to find influential users by exploiting the contents of the messages posted by them to express opinions on items, by modeling these contents with a three-layer network.

The [full paper](http://staff.icar.cnr.it/pizzuti/pubblicazioni/IEEETM2017.pdf) and [other materials](http://staff.icar.cnr.it/pizzuti/codice/SocialAU/readme.html) are avaible on [*ICAR-CNR*](https://www.icar.cnr.it/) website.

I try to develop an algorithm like **SocialAU** using [**JAX**](https://github.com/google/jax), [**PyTorch**](https://pytorch.org/) and [**NumPy**](https://numpy.org/) for calculations between tensors.

<br>

![social-network](https://www.icar.cnr.it/wp-content/uploads/2018/01/SocialCommerce.png)

<br>

[Image Credits](https://www.icar.cnr.it/progetti/social-commerce/)

<br>

## Installation

### Dependencies

```
* Python >= 3.8.5
* Numpy >= 1.20
* PyTorch >= 1.10.2
```

### User Installation

1. Clone or download .zip and unzip
2. Using terminal go into the folder with setup.py
3. Digit the following command
```
python setup.py install
```
4. Try
```
import influencer
influencer.__version__
```

**or**

```
sudo apt install git

pip install git+https://github.com/nickprock/influencer.git

```
## Citation

If you use my code in your research, please cite this project:
```
@misc{influencer-centrality,
  author =       {Nicola Procopio,
  title =        {Influencer Centrality},
  howpublished = {\url{https://github.com/nickprock/influencer}},
  year =         {2020}
}
```
and this paper:
```
@article{oro2017detecting,
  title={Detecting topic authoritative social media users: a multilayer network approach},
  author={Oro, Ermelinda and Pizzuti, Clara and Procopio, Nicola and Ruffolo, Massimo},
  journal={IEEE Transactions on Multimedia},
  volume={20},
  number={5},
  pages={1195--1208},
  year={2017},
  publisher={IEEE}
}
```

## Other Functions

The package contains others centrality measure like:
* [HITS and TOPHITS](https://en.wikipedia.org/wiki/HITS_algorithm)

## Test

I tested the performance about **JAX** vs **Numpy** vs **PyTorch** on *HITS* algorithm [here on Google Colab](https://colab.research.google.com/drive/1q4hpkp1Wqb7qEZIY6_EgHBaOt5E3zp6i?usp=sharing).
Other tests are available in [notebook directory](https://github.com/nickprock/influencer/tree/master/notebook).

<br>

![exe_time](https://github.com/nickprock/influencer/blob/master/img/exeTime.png)

<br>

The focus on ***Numpy vs PyTorch*** 

<br>

![focus_on](https://github.com/nickprock/influencer/blob/master/img/focus.png)

<br>

Execution time in **log10 scale**. 

<br>

![focus_on](https://github.com/nickprock/influencer/blob/master/img/exeTimeLog.png)

<br>

At the moment numpy work better than JAX but I may have made some mistakes (the reason could be [this](https://stackoverflow.com/questions/51177788/cupy-is-slower-than-numpy)).
PyTorch is the best implementation, in socialAU works fine up to 10^9 nodes, runtime broke down with a tensor of about 5B nodes. Numpy stops at 10^8 nodes.

**Please report it in the issues.**


**The library allows you to use only numpy versions beacuse [JAX is not avaible for Windows](https://github.com/google/jax#installation), but the script is into *lazy_centrality.py***

## Links

* [JAX, aka NumPy on steroids](https://iaml.it/blog/jax-intro-english)
* [Google researchers introduce JAX: ...](https://hub.packtpub.com/google-researchers-introduce-jax-a-tensorflow-like-framework-for-generating-high-performance-code-from-python-and-numpy-machine-learning-programs/)
* [You don't know JAX](https://colinraffel.com/blog/you-don-t-know-jax.html)


 **Have Fun!**

License
---
The code present in this project is licensed under the MIT License.

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Licenza Creative Commons" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />Quest'opera è distribuita con Licenza <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribuzione 4.0 Internazionale</a>.
