from skimage.color import rgb2lab, lab2rgb

for lab_img, img, reference_img in train_generator:
  break

lab_img[:, :, :, 0].numpy().max(), lab_img[:, :, :, 0].numpy().min()

lab_img.shape

x_rgb = lab_to_rgb(lab_img)
x_rgb.shape

x_rgb.numpy().max()

# https://fairyonice.github.io/Color-space-defenitions-in-python-RGB-and-LAB.html

def extract_single_dim_from_LAB_convert_to_RGB(image, idim):
  '''
  image is a single lab image of shape (None,None,3)
  '''
  z = np.zeros(image.shape)
  if idim != 0 :
      z[:,:,0]=80 ## I need brightness to plot the image along 1st or 2nd axis
  z[:,:,idim] = image[:,:,idim]
  z = lab2rgb(z)
  return(z)

count = 1
fig = plt.figure(figsize=(13,3))

ax = fig.add_subplot(1,3,count)
lab_rgb_gray = extract_single_dim_from_LAB_convert_to_RGB(lab,0) 
ax.imshow(lab_rgb_gray); ax.axis("off")
ax.set_title("L: lightness")
count += 1

ax = fig.add_subplot(1,3,count)
lab_rgb_gray = extract_single_dim_from_LAB_convert_to_RGB(lab,1) 
ax.imshow(lab_rgb_gray); ax.axis("off")
ax.set_title("A: color spectrums green to red")
count += 1

ax = fig.add_subplot(1,3,count)
lab_rgb_gray = extract_single_dim_from_LAB_convert_to_RGB(lab,2) 
ax.imshow(lab_rgb_gray); ax.axis("off")
ax.set_title("B: color spectrums blue to yellow")
count += 1
plt.show()

count = 1
fig = plt.figure(figsize=(13,3))

ax = fig.add_subplot(1,3,count)
lab_rgb_gray = extract_single_dim_from_LAB_convert_to_RGB(lab,0) 
ax.imshow(lab_rgb_gray); ax.axis("off")
ax.set_title("L: lightness")
count += 1

ax = fig.add_subplot(1,3,count)
lab_rgb_gray = extract_single_dim_from_LAB_convert_to_RGB(lab,1) 
ax.imshow(lab_rgb_gray); ax.axis("off")
ax.set_title("A: color spectrums green to red")
count += 1

ax = fig.add_subplot(1,3,count)
lab_rgb_gray = extract_single_dim_from_LAB_convert_to_RGB(lab,2) 
ax.imshow(lab_rgb_gray); ax.axis("off")
ax.set_title("B: color spectrums blue to yellow")
count += 1
plt.show()

plt.imshow(lab_to_rgb[0])

plt.imshow(reference_img[0])

plt.imshow(img[0])

