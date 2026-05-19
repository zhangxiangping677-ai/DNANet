import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

 # 加载单张图像
img_path = './dataset/NUDT-SIRST/images/000605.png'  # 这里替换为您的图像路径
img = Image.open(img_path)

# 对图像应用预处理变换
img_tensor = input_transform(img)

# 如果需要将图像转为批次形式（即单张图像的批次），可以使用 unsqueeze
image = img_tensor.unsqueeze(0)  # 添加一个维度，形成(batch_size, channels, height, width)
# 创建一个二维数据数组，例如一个高斯函数
gray_image = 0.2989 * image[0, 0, :, :] + 0.5870 * image[0, 1, :, :] + 0.1140 * image[0, 2, :, :]
gray_image = gray_image.numpy()  # 转换为 numpy 数组


# 绘制灰度等高线图
plt.figure(figsize=(6, 6))
cp = plt.contour(gray_image, cmap='gray')
plt.colorbar(cp)
plt.title("Grayscale Contour Map - Grayscale Image")
plt.xlabel("Width")
plt.ylabel("Height")
plt.show()