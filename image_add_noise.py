# rewrite the code to use opencv
import cv2
import numpy as np
image = cv2.imread("image.png", cv2.IMREAD_COLOR)

# add noise to the image
betas = np.linspace(0, 1, 20)
alphas = 1 - betas
alphas_cumprod = np.cumprod(alphas)
sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
for i in range(20):
    image = image.astype(np.float32) / 255.0
    noise = np.random.randn(*image.shape) * 0.1
    image = image * sqrt_alphas_cumprod[i] + noise * (1 - sqrt_alphas_cumprod[i])
    image = image * 255.0
    image = image.astype(np.uint8)
    cv2.imwrite(f"image_with_noise_{i}.png", image)