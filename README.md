# IKR-Net
 IKR-Net Blind Image Superresolution

This is the repository for the following paper. Please cite this journal paper, if you use this code in your research:

Ates, H. F., Yildirim, S., & Gunturk, B. K. (2023). Deep learning-based blind image super-resolution with iterative kernel reconstruction and noise estimation. Computer Vision and Image Understanding, 233, 103718.

# Required libraries

Required libraries are provided in requirements.txt file.
 
# Test code
You can test the model (for X4 scaling and noise-free images) as follows:

./codes/python test_BLDKernet.py

Test results (PSNR, SSIM, Visuals of SR images and estimated kernels) are provided under "experiments" folder. 

# Trained Models
Trained model should be placed in "models" folder
