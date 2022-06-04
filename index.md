## Poisson2Sparse: Self-Supervised Poisson Denoising From a Single Image

### Abstract
Image enhancement approaches often assume that the noise is signal independent, and approximate the degradation model as zero-mean additive Gaussian noise. However, this assumption does not hold for biomedical imaging systems where sensor-based sources of noise are proportional to signal strengths, and the noise is better represented as a Poisson process. In this work, we explore a sparsity and dictionary learning-based approach and present a novel self-supervised learning method for single-image denoising where the noise is approximated as a Poisson process, requiring no clean ground-truth data. Specifically, we approximate traditional iterative optimization algorithms for image denoising with a recurrent neural network which enforces sparsity with respect to the weights of the network. Since the sparse representations are based on the underlying image, it is able to suppress the spurious components (noise) in the image patches, thereby introducing implicit regularization for denoising task through the network structure. Experiments on two bio-imaging datasets demonstrate that our method outperforms the state-of-the-art approaches in terms of PSNR and SSIM. Our qualitative results demonstrate that, in addition to higher performance on standard quantitative metrics, we are able to recover much more subtle details than other compared approaches.

### Code
Coming Soon!


