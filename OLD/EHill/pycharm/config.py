import albumentations as A


BATCH_SIZE = 30
IMG_SIZE = 160

CLASSES = 1  # number of output channels

EPOCHS = 50


# With extra padding & random crops
AUGMENTER_TRAIN = A.Compose(
    transforms=[
        A.Resize(height=IMG_SIZE, width=IMG_SIZE, always_apply=True, p=1.0),
        # A.RandomCrop(height=IMG_SIZE, width=IMG_SIZE, always_apply=True, p=1.0),

        A.Transpose(p=0.5),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),

    ],
)


AUGMENTER_VAL = A.Compose(
    transforms=[
        A.Resize(height=IMG_SIZE, width=IMG_SIZE, always_apply=True, p=1.0),
        # A.CenterCrop(height=IMG_SIZE, width=IMG_SIZE, always_apply=True, p=1.0),
    ],
)
