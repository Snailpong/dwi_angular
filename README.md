# dwi_angular

PyTorch implementation of the paper "Super-Resolved q-Space Deep Learning" (MICCAI 2019, MedIA 2021)

## Dependencies

- Pytorch
- numpy
- nibabel
- sklearn_extra

## Usage

1. Clone the repository

- `git clone https://github.com/Snailpong/dwi_angular.git`

2. Dataset download

- Download from Human Connectome Project (HCP) Young Adult Database

- Download `data.nii.gz, nodif_brain_mask.nii.gz, bval, bvec` files for each subject

3. Preprocess data

- `python vector_scatter.py`: select index for extracting LR dimension (diffusion)

- `python preprocess.py`: make h5 file for training and testing

4. Train

- `python train.py`

- arguments

  - load_model: `True`/`False`

  - cuda_visible: `CUDA_VISIBLE_DEVICES` (e.g. 1)

  - batch_size: set batch size


5. Test

- `python test.py`

- arguments

  - load_model: `True`/`False`
  
  - cuda_visible: `CUDA_VISIBLE_DEVICES` (e.g. 1)

  - image_path: set validation path