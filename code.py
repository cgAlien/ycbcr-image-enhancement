import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from skimage import img_as_float, exposure, color, io
import tkinter as tk
from tkinter import filedialog
matplotlib.rcParams['font.size'] = 8
def plot_img_and_hist(image, axes, bins=256): """khitam haleem."""
image = img_as_float(image)
ax_img, ax_hist = axes
ax_cdf = ax_hist.twinx()
ax_img.imshow(image, cmap=plt.cm.gray)
ax_img.set_axis_off()
ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
ax_hist.set_xlabel('Pixel intensity')
ax_hist.set_xlim(0, 1)
ax_hist.set_yticks([])
img_cdf, bins = exposure.cumulative_distribution(image, bins)
ax_cdf.plot(bins, img_cdf, 'r')
ax_cdf.set_yticks([])
return ax_img, ax_hist, ax_cdf
def rgb_to_ycbcr(image):
ycbcr_image = color.rgb2ycbcr(image)
return ycbcr_image
def process_image(image_path):
img = io.imread(image_path)
img_ycbcr = rgb_to_ycbcr(img)

if img.ndim == 3:
img = color.rgb2gray(img)
p2, p98 = np.percentile(img, (2, 98))
img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
img_eq = exposure.equalize_hist(img)
img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
fig = plt.figure(figsize=(8, 5))
axes = np.zeros((2, 4), dtype=np.object_)
axes[0, 0] = fig.add_subplot(2, 4, 1)
for i in range(1, 4):
axes[0, i] = fig.add_subplot(2, 4, 1+i, sharex=axes[0, 0], sharey=axes[0, 0])
for i in range(0, 4):
axes[1, i] = fig.add_subplot(2, 4, 5+i)
ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, 0])
ax_img.set_title('Original Image')
ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_rescale, axes[:, 1])
ax_img.set_title('Contrast Stretching')
ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_eq, axes[:, 2])
ax_img.set_title('Histogram Equalization')
ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_adapteq, axes[:, 3])
ax_img.set_title('Adaptive Equalization')
fig.tight_layout()
plt.show()
def open_file_dialog():
file_path = filedialog.askopenfilename(title='Select an Image', filetypes=[('Image
Files', '*.jpg *.jpeg *.png *.bmp *.tif *.tiff')])
if file_path:
process_image(file_path)

# GUI setup
root = tk.Tk()
root.title("Image Processing")
# Set window size
root.geometry("600x400") # Width x Height
select_image_btn = tk.Button(root, text="Select Image", command=open_file_dialog, padx=20, pady=10)
select_image_btn.pack()
root.mainloop()
