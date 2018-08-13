# python-GLLIM

Python version of *GLLiM* algorithm, written during an internship at Inria (Grenoble, 2018).

This repository exposes two major features :
- A rather succinct implementation of *GLLiM* 
- Severals tools to apply *GLLiM* on inverse problem

## Organisation

- The actual *GLLiM* implementation can be found in [Core.gllim](Core/gllim.py) package. 
Three variantes are provided : [dGLLiM](Core/dgllim.py), [jGLLiM](Core/gllim.py), [sGLLiM](Core/sGllim.py).


- The package [tools](tools) provides ways to generate data (via [`tools.context.abstractContext`](tools/context.py)), 
save and load data and models ([`tools.archive`](tools/archive.py)),
and measure quality of estimation ([`tools.measures`](tools/measures.py)). 
The visualization of the results is facilitated by [`tools.results`](tools/results.py).
The class [`tools.experience.Experience`](tools/experience.py) wraps all theses helpers.

- The package [plotting](plotting) draws severals graphs and animations (used by [`Experience`](tools/experience.py)).

- The main application of the internship was to invert the Hapke model, which is implemented in [hapke](hapke) package.

- Finally, simulations on the Hapke model were carried out with the [experiences](experiences) package.


## Requirements
Requirements can be found [here](requirements.txt).

## Related package 
You can found another Python implementation [here](https://github.com/Chutlhu/pyGLLiM).
