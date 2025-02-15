# TINST-Net: A Novel Neural Style Transfer using Mixture of Text and Image
*Accepted to MAPR 2024*

![Project Image](images/styletransfer)

TINST-Net is a novel framework for neural style transfer that leverages both text descriptions and reference images to create artistic transformations. It combines the power of **CLIP**, **VGG-19**, and **U-Net** to achieve refined style transfer and creative style blending.

## Installation

### Requirements
- Python 3.6 or higher
- PyTorch
- torchvision
- matplotlib
- Pillow

### Setup Instructions

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/tinstnet.git
   cd tinstnet
2. Create a virtual environment and activate it:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
3. Install the required packages
   ```sh
   pip install -r requirements.txt
4. Ensure that the paths in config.py are correctly set to your content image and desired output directory
5. Run the main script
   ```sh
   python main.py

### Acknowledgments
This project is heavily adapted from:  
- [CLIPstyler](https://github.com/cyclomon/CLIPstyler)  
- [PyTorch Neural Style Transfer Tutorial](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)  

## Citation

If you find this work useful, please cite our paper:
```bibtex
@inproceedings{tran2024tinstnet,
  booktitle={2024 International Conference on Multimedia Analysis and Pattern Recognition (MAPR)}, 
  title={TINST-Net: A Novel Neural Style Transfer using Mixture of Text and Image}, 
  year={2024},
  volume={},
  number={},
  pages={1-6},
  doi={10.1109/MAPR63514.2024.10660938}}

