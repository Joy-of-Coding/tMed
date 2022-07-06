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
