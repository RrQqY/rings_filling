import cv2
import matplotlib.pyplot as plt

# 以灰度模式读取图像
image = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)

# 二值化
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

# 腐蚀
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
binary = cv2.erode(binary, kernel)

# 应用Canny边缘检测
edges = cv2.Canny(binary, 100, 200)
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# 显示原始图像和结果
plt.subplot(131), plt.imshow(image, cmap='gray'), plt.title('原始图像')
plt.subplot(132), plt.imshow(edges, cmap='gray'), plt.title('Canny边缘检测')

# 显示轮廓contours
cv2.drawContours(image, contours, -1, (0, 0, 255), 2)
plt.subplot(133), plt.imshow(image, cmap='gray')
# 打印轮廓数量
print("len", len(contours))

plt.show()
