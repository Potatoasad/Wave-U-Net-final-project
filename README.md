# Wave U Net architecture for music source seperation | CSE 382M final project

Provided here is code implementing the Wave-U-Net architecture in pytorch for the purposes of music source seperation using the [MUSDB dataset](https://sigsep.github.io/datasets/musdb.html#musdb18-compressed-stems)\. 
The architecture was introduced in https://arxiv.org/abs/1806.03185 for the same purpose using the same dataset, and we differ in our setup from the authors only in some minor differences in training techniques and loss functions and more (as we experiment). 

It is a work in progress and we will be using this repository to run tests and report results. 

This code has been implemented for the CSE382M "Foundations of Data Science" final project. There is a more thorough version of this architecture at https://github.com/f90/Wave-U-Net-Pytorch which is coded in a more modular and useful fashion and the only thing that this repository really __may__ do better is that it runs on the MPS backend (Mac M1 GPU) out the box. 

## Usage
This repository gives everyone a:
1. A self contained environment in which you can run notebooks
2. Allows you to work on a package inside the `\scripts` folder, where you can save functions/classes you want to use inside the jupyter notebook (this keeps your notebook relatively clean and free of a cascade of initial function defintions)

#### Initial Install
First add whatever packages you will need inside the `requirements.txt`

Then:
```bash
make install  #installs everything and creates the environment
```
You only need to do this once (or whenever you add another package to `requirements.txt`)

#### Running the notebook
Once installed, you can simply run:

```bash
./jupyter lab # launches jupyter lab in that environment, you could do ./jupyter notebook too
```

Then in the jupyter notebook you can do whatever you want.

#### Saving useful functions and classes
Add any classes or functions to the `/scripts` folder as you would in a package. 
Then in the notebook you can do:
```python
from scripts import *
```
and all the functions and classes will be available to you inside the notebook. This keeps your notebook relatively clean and free of a cascade of initial function defintions. 

Simply restart the notebook kernel and any changes you have made to `/scripts` will be updated. 

