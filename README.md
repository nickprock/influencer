# Influencer Centrality

The algorithm in this package is inspired by the paper **"Detecting Topic Authoritative Social Media Users: a Multilayer Network Approach"**.

In this paper, the authors propose a method capable of finding influential users by exploiting the contents of the messages posted by them to express opinions on items, by modeling these contents with a three-layer network.

The [full paper](http://staff.icar.cnr.it/pizzuti/pubblicazioni/IEEETM2017.pdf) and [other materials](http://staff.icar.cnr.it/pizzuti/codice/SocialAU/readme.html) are available on [*ICAR-CNR*](https://www.icar.cnr.it/) website.

This package provides a PyTorch-based implementation of the **SocialAU** algorithm for efficient tensor calculations.

<br>

![social-network](https://www.icar.cnr.it/wp-content/uploads/2018/01/SocialCommerce.png)

<br>

[Image Credits](https://www.icar.cnr.it/progetti/social-commerce/)

<br>

## Installation

### Dependencies

```
* Python >= 3.8.5
* PyTorch >= 1.10.2
* NumPy >= 1.20
```

### User Installation

**Option 1: Clone and install locally**
1. Clone or download .zip and unzip
2. Using terminal go into the folder with setup.py
3. Run the following command:
```bash
python setup.py install
```
4. Test the installation:
```python
import influencer
print(influencer.__version__)
```

**Option 2: Install from GitHub**
```bash
pip install git+https://github.com/nickprock/influencer.git
```

## Current Implementation

The current main branch contains a stable PyTorch-based implementation focused on:
- **SocialAU algorithm** for detecting influential users
- **HITS and TOPHITS** centrality measures
- Comprehensive test suite
- Optimized performance for large-scale networks

### Experimental Features

For experimental features including JAX and NumPy implementations, please check the `experimental` branch:
```bash
git clone https://github.com/nickprock/influencer.git
git checkout experimental
```

## Features

The package includes the following centrality measures:
* **SocialAU**: Multi-layer network approach for topic-authoritative user detection
* **[HITS and TOPHITS](https://en.wikipedia.org/wiki/HITS_algorithm)**: Hub and Authority scoring algorithms

## Performance

The PyTorch implementation provides excellent performance characteristics:
- **Scalability**: Works efficiently up to 10^9 nodes
- **Memory efficiency**: Optimized tensor operations
- **Cross-platform compatibility**: Works on Windows, Linux, and macOS

For detailed performance comparisons between different implementations (JAX, NumPy, PyTorch), refer to the `experimental` branch and the [Google Colab notebook](https://colab.research.google.com/drive/1q4hpkp1Wqb7qEZIY6_EgHBaOt5E3zp6i?usp=sharing).

Additional tests and examples are available in the [notebook directory](https://github.com/nickprock/influencer/tree/master/notebook).

## Testing

The package includes a comprehensive test suite to ensure reliability and correctness of the algorithms. Run tests with:
```bash
python -m pytest tests/
```

## Citation

If you use this code in your research, please cite this project:
```bibtex
@misc{influencer-centrality,
  author = {Nicola Procopio},
  title = {Influencer Centrality},
  howpublished = {\url{https://github.com/nickprock/influencer}},
  year = {2020}
}
```

and the original paper:
```bibtex
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

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

For experimental features or alternative implementations, consider contributing to the `experimental` branch.

## Links

* [JAX, aka NumPy on steroids](https://iaml.it/blog/jax-intro-english)
* [Google researchers introduce JAX](https://hub.packtpub.com/google-researchers-introduce-jax-a-tensorflow-like-framework-for-generating-high-performance-code-from-python-and-numpy-machine-learning-programs/)
* [You don't know JAX](https://colinraffel.com/blog/you-don-t-know-jax.html)

**Have Fun!**

## License

The code present in this project is licensed under the MIT License.

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution 4.0 International License</a>.