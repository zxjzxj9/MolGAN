# MolGAN
MolGAN Implementation with PyTorch, using RDKit as molecule processing tool.

## Prerequisites
PyTorch == 1.8.1

RDKit == 2020.09.1.0

## Reference Paper
[MolGAN: An implicit generative model for small molecular graphs](https://arxiv.org/pdf/1805.11973.pdf)

## Dataset
[QM9](https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904)

Remember to decompress the bzip2 file to tar file using bunzip2 command instead. 
The code uses xyz coordinate file as molecule input format

## Training
Modify the hyper parameters using hparam.toml file. Just using the command `python train.py` to start training.

## Inference
The inference code is in `inferenc.py`, we can launch an inference running by the following command, 
this command will print out *batch_size* size of SMILES strings of the generated molecules.
```python infer.py -c hparam.toml -b [batch_size] -m model.pt```