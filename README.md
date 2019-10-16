# Introduction
This is the pytorch implementation of the paper "Parallel Attention: A Unified Framework for Visual Object Discovery through Dialogs and Queries"


***If you use this code in your research, please cite our paper:***

```
@article{zhuang2017parallel,
  title={Parallel Attention: A Unified Framework for Visual Object Discovery through Dialogs and Queries},
  author={Zhuang, Bohan and Wu, Qi and Shen, Chunhua and Reid, Ian and Hengel, Anton van den},
  journal={arXiv preprint arXiv:1711.06370},
  year={2017}
}
```

## Dataset
Please download the MSCOCO dataset and the GuessWhat?! annotations:

* MSCOCO dataset http://cocodataset.org/#home

* GuessWhat?! dataset https://github.com/GuessWhatGame/guesswhat


## Code

__utils.py__: provide necessary functions  
__main.py__: main file, implementing training and testing          
__read_data.py__: self-defined data layer  
__config.yaml__: define the necessary hyperparameters (e.g., data directory, GPU), please modify this file  
**model.py**: define the whole framework    
**./modules/**: defines the attention module and the LSTM module    
**./data/**: provide necessary data used in our experiment    
  

## Training

```
python main.py

```


## Copyright

Copyright (c) Bohan Zhuang. 2017

** This code is for non-commercial purposes only. For commerical purposes,
please contact Chunhua Shen <chhshen@gmail.com> **

This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

