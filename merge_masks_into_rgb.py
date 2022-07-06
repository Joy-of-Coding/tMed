import cv2
import numpy as np
import matplotlib.pyplot as plt

def merge_masks_into_rgb(list_of_masks):
  
  output = np.zeros((224, 224, 3))
  mask_count = 0

  for mask in list_of_masks:
      for height in range(len(mask)): 
        for pixel in range(len(mask[height])):
          if mask[height][pixel] == 1 or mask[height][pixel] == 255:
              output[height][pixel][mask_count] = 255
      mask_count += 1

  return output

# Load image as array
input_dir = 'data/Training_Input/pigment_network/'
pigment_dir = 'data/Training_GroundTruth/pigment_network/'
lesion_dir = 'data/Training_GroundTruth/lesions/pigment_network/'

input = cv2.imread(input_dir + sorted(os.listdir(input_dir))[0])
plt.imshow(input)
plt.show()

lesion = cv2.imread(lesion_dir + sorted(os.listdir(lesion_dir))[0], 0)
lesion = cv2.resize(lesion, (224, 224))
plt.imshow(lesion)
plt.show()

pigment = cv2.imread(pigment_dir + sorted(os.listdir(pigment_dir))[0], 0)
pigment = cv2.resize(pigment, (224, 224))
plt.imshow(pigment)
plt.show()
