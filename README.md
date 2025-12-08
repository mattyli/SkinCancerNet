## Links
- https://derma.jmir.org/2022/3/e39143/

# Pipeline
## Pre Processing
- remove ruler lines from images
- remove hair from images
    - https://pdf.sciencedirectassets.com/280281/1-s2.0-S1110866525X00049/1-s2.0-S1110866525002373/main.pdf
        - use a VAE to remove the hair?
            - https://pmc.ncbi.nlm.nih.gov/articles/PMC9907627/pdf/SRT-28-445.pdf
            - L_(SSIM) + L_2 loss produces the best results
            - Encoder Architecture
                - Layer 1
                    - Kernel Size = 3
                    - Activation = ReLU
                    - Stride = 2
                    - Num Filters = 64 (number of kernels)
                - Layer 2
                    - Kernel Size = 3
                    - Activation = ReLU
                    - Stride = 2
                    - Num Filters = 32 (number of kernels)
                - Flatten
                - Compute mean and variance for the distribution, latent dimension = 50
                - Sample from the distribution
                - unflatten 
            - Decoder Architecture
                - Layer 1
        - DullRazor Algorithm? (Hair Density Score)
            - Target size (images resized to 256x256)
            - Kernel = 7x7
            - Split at `score=0.015` yields 8423 hairy samples, 5596 clean
                - take the bottom 1000 clean images
        - CycleGANs? --> some kind of style transfer
- random crop and rotation
- style transfer to get dark skin
- Challenges
    - hair comes in different color, shapes, textures and density, classical CV can only do so much
    - 