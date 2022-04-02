### This repo contains the code for my class project in CS236. 

I extend the code of the "default" project provided by the teaching staff (https://github.com/deepgenerativemodels/default-project) and implement three models to discover interpretable GAN control: 

1) Find PCA directions as proposed in 

E. Härkönen, A. Hertzmann, J. Lehtinen, and S. Paris, “Ganspace: Discovering interpretable gan
controls,” arXiv preprint arXiv:2004.02546, 2020.

2) Find interpretable directions with reconstrcutor: 

A. Voynov and A. Babenko, “Unsupervised discovery of interpretable directions in the gan latent
space,” in International Conference on Machine Learning, pp. 9786–9796, PMLR, 2020.

3) Acquire interpretable GAN control by training an InfoGAN: 

X. Chen, Y. Duan, R. Houthooft, J. Schulman, I. Sutskever, and P. Abbeel, “Infogan: Interpretable
representation learning by information maximizing generative adversarial nets,”

### Examples of directions found that correspond to interpretable changes in the images

change of dog-color: 
![image](https://user-images.githubusercontent.com/90799324/161369342-d734d56a-7d13-4647-9704-cb9de0f3c515.png)

change of dog (head) size: 
![image](https://user-images.githubusercontent.com/90799324/161369354-2050478a-5938-4808-a65d-434f066611b0.png)

change of image background: 
![image](https://user-images.githubusercontent.com/90799324/161369419-e213723b-c76b-473a-bcf8-c40b28acec62.png)
