import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy import ndimage

def process_image(image):
    # 轉換為灰度圖像
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)
    
    # 執行2D傅立葉轉換
    fft2 = fftpack.fft2(image)
    fft2_shifted = fftpack.fftshift(fft2)
    
    # 創建高通濾波器（保留邊緣細節）
    rows, cols = image.shape
    crow, ccol = rows//2, cols//2
    mask = np.ones((rows, cols))
    mask[crow-30:crow+30, ccol-30:ccol+30] = 0  # 高通濾波
    
    # 創建低通濾波器（模糊效果）
    mask_low = np.zeros((rows, cols))
    mask_low[crow-30:crow+30, ccol-30:ccol+30] = 1  # 低通濾波
    
    # 應用濾波器
    fft2_high = fft2_shifted * mask
    fft2_low = fft2_shifted * mask_low
    
    # 反變換
    image_high = np.real(fftpack.ifft2(fftpack.ifftshift(fft2_high)))
    image_low = np.real(fftpack.ifft2(fftpack.ifftshift(fft2_low)))
    
    # 顯示頻譜（取對數以便觀察）
    spectrum = np.log(np.abs(fft2_shifted) + 1)
    
    return image, spectrum, image_high, image_low

# 創建一個模擬的測試圖像（因為我們沒有真實圖片輸入）
def create_test_image():
    # 創建一個包含不同特徵的測試圖像
    x, y = np.meshgrid(np.linspace(-5, 5, 256), np.linspace(-5, 5, 256))
    
    # 添加一些圓形和方形圖案
    circle = (x**2 + y**2 <= 4)
    square = (np.abs(x) <= 2) & (np.abs(y) <= 2)
    
    # 添加一些紋理
    texture = np.sin(x*2) * np.cos(y*2)
    
    # 組合圖案
    image = circle.astype(float) + square.astype(float) * 0.5 + texture * 0.2
    
    return image

# 生成並處理圖像
test_image = create_test_image()
original, spectrum, high_pass, low_pass = process_image(test_image)

# 顯示結果
plt.figure(figsize=(15, 10))

plt.subplot(221)
plt.imshow(original, cmap='gray')
plt.title('原始圖像')

plt.subplot(222)
plt.imshow(spectrum, cmap='gray')
plt.title('頻譜')

plt.subplot(223)
plt.imshow(high_pass, cmap='gray')
plt.title('高通濾波（邊緣增強）')

plt.subplot(224)
plt.imshow(low_pass, cmap='gray')
plt.title('低通濾波（模糊效果）')

plt.tight_layout()
plt.show()