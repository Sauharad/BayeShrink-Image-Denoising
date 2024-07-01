import pywt
import numpy as np
import cv2
import matplotlib.pyplot as plt

def add_gaussian_noise(image, mean=0, std=40):
    # Generate Gaussian noise
    gaussian_noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + gaussian_noise
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)

def bayes_shrink_denoise(image, wavelet='haar'):
    coeffs = pywt.wavedec2(image, wavelet)
    
    # Estimate the noise variance from detail coefficients
    detail_coeffs = coeffs[-1]
    noise_var = np.median(np.abs(detail_coeffs)) / 0.6745
    noise_var = noise_var ** 2
    
    def bayes_threshold(coeffs, noise_var):
        var_coeffs = np.var(coeffs)
        if var_coeffs == 0:
            return coeffs  # No thresholding if variance is zero
        
        # Calculate the BayesShrink threshold
        threshold = noise_var / np.sqrt(var_coeffs)
        return pywt.threshold(coeffs, threshold, mode='soft')
    
    coeffs[1:] = [tuple(bayes_threshold(c, noise_var) for c in detail) for detail in coeffs[1:]]
    denoised_image = pywt.waverec2(coeffs, wavelet)
    denoised_image = np.clip(denoised_image, 0, 255)
    
    return denoised_image.astype(np.uint8)

image = cv2.imread(r'C:\Users\Shaur\OneDrive\Desktop\lena.png', cv2.IMREAD_GRAYSCALE)

noisy_image = add_gaussian_noise(image)

denoised_image = bayes_shrink_denoise(noisy_image)

plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Noisy Image')
plt.imshow(noisy_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Denoised Image')
plt.imshow(denoised_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()