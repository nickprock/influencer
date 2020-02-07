from setuptools import setup

exec(open('influencer/version.py').read())
setup(
   name='influencer',
   version=__version__,
   description='A python library to find influencer in a social network',
   license="LICENSE.txt",
   long_description=open('README.md').read(),
   author='Nicola Procopio',
   url="https://github.com/nickprock/",
   packages=['influencer'],  #same as name
   # install_requires=['numpy', 'jax'],
   install_requires=['numpy'], #external packages as dependencies
)
