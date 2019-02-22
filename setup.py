from setuptools import setup

with open("README.md") as f:
   readme = f.read()

setup(
     name='particle_filter',    # This is the name of your PyPI-package.
     version='0.0.0',                          # Update the version number for new releases
     packages=['particle_filter'],                  # The name of your scipt, and also the command you'll be using for calling it
     description = 'A basic particle filter',
     author = 'Kayla Comalli',
     long_description_content_type="text/markdown",
     long_description=readme,
     author_email = 'kayla.comalli@gmail.com',
     url = 'https://github.com/kalyco/particle_filter', # use the URL to the github repo
    keywords=["particle", "probabilistic", "stochastic", "filter", 
    "filtering"]
 )