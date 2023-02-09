# Unpaired Image-to-Image Translation through CycleGAN
***The term project of COMP511: Computer Vision with Deep Learning in Fall 2022***
The CycleGAN model is built based on [CycleGAN](https://github.com/junyanz/CycleGAN) by Jun-Yan Zhu et al. Training backlogs are available in Weights & Biases.    

General architecture of the CYCLEGAN model:

<img width="494" alt="image" src="https://user-images.githubusercontent.com/60810553/217907341-10cdc9c7-7e26-44fb-afe4-dac60ca279c7.png">

## The difference between the CycleGAN and base CycleGAN:  

### **Generator:**
- There are nine residual blocks.
- LeakyReLU is implemented instead of ReLU.

<img width="476" alt="image" src="https://user-images.githubusercontent.com/60810553/217908391-a7603100-7879-41d8-8f89-f4e26ea7a48a.png">

### **Discriminator:**
- Instance normalization is implemented after each convolutional layer.

<img width="456" alt="image" src="https://user-images.githubusercontent.com/60810553/217908938-015010ac-d72c-4f12-8870-32fdf2a4bba9.png">
