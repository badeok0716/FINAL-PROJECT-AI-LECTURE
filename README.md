# Text Augmentation method for Text GAN methods. (The final project of AI special lecture in SNU)
Note that codes are adapted from https://github.com/williamSYSU/TextGAN-PyTorch. 

## Requirements

- **PyTorch >= 1.1.0**
- Python 3.6
- Numpy 1.14.5
- CUDA 7.5+ (For GPU)
- nltk 3.4
- tqdm 4.32.1
- KenLM (https://github.com/kpu/kenlm)

## KenLM Installation

- Download stable release and unzip: http://kheafield.com/code/kenlm.tar.gz

- Need Boost >= 1.42.0 and bjam

  - Ubuntu: `sudo apt-get install libboost-all-dev`
  - Mac: `brew install boost; brew install bjam`

- Run *within* kenlm directory:

  ```bash
  mkdir -p build
  cd build
  cmake ..
  make -j 4
  ```

- `pip install https://github.com/kpu/kenlm/archive/master.zip`

- For more information on KenLM see: https://github.com/kpu/kenlm and http://kheafield.com/code/kenlm/

## Run experiment

- To run SeqGAN
  ```python
  cd run_scripts
  python run_seqgan.py --train_ratio {train ratio} --trial {trial} --aug {augmentation type}
  ```

- To run RelGAN
  ```python
  cd run_scripts
  python run_relgan.py --train_ratio {train ratio} --trial {trial} --aug {augmentation type}
  ```

* train ratio - float<br />
    the proportion of training examples used to train GAN model<br />
    default - 1.0
    e.g.) 0.05, 1.0, ...
* trial - int, <br />
    the index of trial (used to the name of exp directory)<br />
    e.g.) 1, 2, 3, ...
* augmentation type - str, <br />
    the augmentation transform apply to update GAN model. (The combination of 'mask', 'rand', 'swap'. Use 'noaug' to train without augmentation)<br />
    e.g.) 'noaug', 'swap', 'mask', 'swap_mask', 'rand_mask_swap', ... <br />
    Note that 'rand_mask_swap' means that the augmentations 'rand', 'mask', and 'swap' apply to a text sequentially.