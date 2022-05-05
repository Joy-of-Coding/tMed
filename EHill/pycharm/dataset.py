
import numpy as np
from tensorflow.keras.preprocessing.image import Iterator


def random_image_and_mask(config):
    S = config.IMG_SIZE
    img = np.zeros(shape=(S, S, 3), dtype=np.uint8)
    mask = np.zeros(shape=(S, S, config.CLASSES), dtype=np.uint8)

    # Random color for img
    img += np.random.randint(0, 255, size=(3,), dtype=np.uint8)

    # Add a random box
    w, h = np.random.randint(int(S * 0.1), int(S * 0.3), size=(2,))
    box_color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)
    x, y = np.random.randint(0, int(S * 0.9), size=(2,))
    img[y:y+h, x:x+w, :] = box_color
    mask[y:y+h, x:x+w, :] = 255

    return img, mask


class IteratorWithAug(Iterator):
    """ Iterator that generate data from directory and a list of images and a
        corresponding list of class labels
    """

    def __init__(self,
                 image_paths,
                 mask_paths,
                 config,
                 augmenter=None,
                 mode=None,
                 shuffle=True,
                 batch_size=16,
                 seed=None,
                 ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.n = len(image_paths)
        assert self.n == len(mask_paths)
        self.batch_size = batch_size
        self.mode = mode
        self.config = config
        self.augmenter = augmenter

        super().__init__(self.n,
                         batch_size,
                         shuffle,
                         seed=seed)

    def _get_batches_of_transformed_samples(self, index_array):
        # build batch of image & gt
        batch_x = []
        batch_y = []
        for i, j in enumerate(index_array):
            # TODO: read rgb from: self.image_paths[j]
            # TODO: read bw mask from: self.mask_paths[j]
            rgb, bw = random_image_and_mask(self.config)

            # Run Augmentation
            if self.augmenter is not None:
                transformed = self.augmenter(image=rgb, mask=bw, keypoints=[])
                transformed_image = transformed['image']
                transformed_mask = transformed['mask']
            else:
                transformed_image = rgb
                transformed_mask = bw

            if len(transformed_mask.shape) == 2:
                transformed_mask = transformed_mask[..., np.newaxis]

            batch_x.append(transformed_image)
            batch_y.append(transformed_mask)

        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)

        # standardize
        batch_x = (batch_x.astype(np.float32) / 255) - 0.5
        batch_y = (batch_y > 127).astype(np.float32)

        return batch_x, batch_y
