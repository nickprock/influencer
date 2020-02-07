# Influencer Centrality

The algorithm in this package is inspired about the paper **"Detecting Topic Authoritative Social Media Users: a
Multilayer Network Approach"**.
In this paper, the authors propose a method capable to find influential users by exploiting the contents of the messages posted by them to express opinions on items, by modeling these contents with a three-layer network.

The [full paper](http://staff.icar.cnr.it/pizzuti/pubblicazioni/IEEETM2017.pdf) and [other materials](http://staff.icar.cnr.it/pizzuti/codice/SocialAU/readme.html) are avaible on [*ICAR-CNR*](https://www.icar.cnr.it/) website.

I try to develop an algorithm like **SocialAU** using [**JAX**](https://github.com/google/jax) and [**NumPy**]() for calculations between tensors.

<br>

![social-network](https://www.icar.cnr.it/wp-content/uploads/2018/01/SocialCommerce.png)

<br>

[Image Credits](https://www.icar.cnr.it/progetti/social-commerce/)

<br>

## Citation

If you use my code in your research, please cite this project:
```
@misc{influencer-centrality,
  author =       {Nicola Procopio,
  title =        {Influencer Centrality},
  howpublished = {\url{https://github.com/nickprock/influencer-centrality}},
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
* [PageRank](https://en.wikipedia.org/wiki/PageRank) [not avaible now]

## Test

I tested the performance about **JAX** on [Google Colab](https://colab.research.google.com/drive/1c7CFyuKYpV8Ngf5RrUAyp1DZf-qYiQVv).

At the moment numpy work better than JAX but I may have made some mistakes (the reason could be [this](https://stackoverflow.com/questions/51177788/cupy-is-slower-than-numpy)). **Please report it in the issues.**


**The library allows you to use both versions.**

## Links

* [JAX, aka NumPy on steroids](https://iaml.it/blog/jax-intro-english)
* [Google researchers introduce JAX: ...](https://hub.packtpub.com/google-researchers-introduce-jax-a-tensorflow-like-framework-for-generating-high-performance-code-from-python-and-numpy-machine-learning-programs/)
* [You don't know JAX](https://colinraffel.com/blog/you-don-t-know-jax.html)


 **Have Fun!**


---

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Licenza Creative Commons" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />Quest'opera è distribuita con Licenza <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribuzione 4.0 Internazionale</a>.
