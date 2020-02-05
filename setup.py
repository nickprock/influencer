from setuptools import setup

exec(open('influencer-centrality/version.py').read())
setup(
   name='influencer-centrality',
   version=__version__,
   description='A python library to find influencer in a social network',
   license="LICENSE.txt",
   long_description=open('README.md').read(),
   author='Nicola Procopio',
   url="https://github.com/nickprock/",
   packages=['influencer-centrality'],  #same as name
   install_requires=['numpy'], #external packages as dependencies
)
