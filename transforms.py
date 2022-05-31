import albumentations as A
import cv2

NORMALIZATION_PARAMS = {
    'color mean': (0.4497, 0.4503, 0.4181),
    'color std': (0.2445, 0.2435, 0.2646),
    'bw mean': (0.4468, 0.4468, 0.4468),
    'bw std': (0.2408, 0.2408, 0.2408)
}


class TransformCollection:
    @staticmethod
    def train_transform():
        pass

    @staticmethod
    def validation_transform():
        pass


class IBotWeakTransformCollection(TransformCollection):
    @staticmethod
    def train_transform(size_: int):
        _normalization = A.Normalize(mean=NORMALIZATION_PARAMS['color mean'], std=NORMALIZATION_PARAMS['color std'])
        train_transform = A.Compose([
            A.augmentations.geometric.resize.Resize(
                height=size_,
                width=size_
            ),
            A.Rotate(
                limit=30,
                p=0.5
            ),
            A.HorizontalFlip(
                p=0.5
            ),
            A.ISONoise(
                color_shift=(0.01, 0.01),
                intensity=(0.01, 0.01),
                always_apply=False,
                p=0.5
            ),
            A.OpticalDistortion(
                distort_limit=0.3,
                shift_limit=0.1,
                interpolation=1,
                border_mode=4,
                value=None,
                mask_value=None,
                always_apply=False,
                p=0.5
            ),
            A.GaussNoise(
                var_limit=(20.0, 100.0),
                mean=0,
                per_channel=True,
                always_apply=False,
                p=0.3
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.1,
                brightness_by_max=False,
                always_apply=False,
                p=0.5
            ),

            _normalization
        ])
        return train_transform

    @staticmethod
    def validation_transform(size_: int):
        _normalization = A.Normalize(mean=NORMALIZATION_PARAMS['color mean'], std=NORMALIZATION_PARAMS['color std'])
        val_transform = A.Compose([
            A.augmentations.geometric.resize.Resize(
                height=size_,
                width=size_
            ),

            _normalization
        ])
        return val_transform



class StrongTransformCollection(TransformCollection):
    @staticmethod
    def train_transform(size_: int):
        _randomrotation_transform = A.augmentations.geometric.rotate.Rotate(
            limit=15,
            interpolation=1,
            border_mode=cv2.BORDER_REFLECT_101,
            value=None,
            mask_value=None,
            always_apply=False,
            p=0.3
        )
        _horizontalflip_transorm = A.augmentations.transforms.HorizontalFlip(p=0.5)
        _opticaldistorsion_transform = A.augmentations.transforms.OpticalDistortion(
            distort_limit=0.4,
            shift_limit=0.1,
            interpolation=1,
            border_mode=cv2.BORDER_REFLECT,
            value=None,
            mask_value=None,
            always_apply=False,
            p=0.1
        )
        _motionblur_transorm = A.augmentations.transforms.MotionBlur(
            blur_limit=(10, 15),
            p=0.2
        )
        _isonoise_transorm = A.augmentations.transforms.ISONoise(
            color_shift=(0.1, 0.2),
            intensity=(0.6, 0.7),
            always_apply=False,
            p=0.1
        )
        _glassblur_transform = A.augmentations.transforms.GlassBlur(
            sigma=0.5,
            max_delta=3,
            iterations=1,
            always_apply=False,
            mode='fast',
            p=0.1
        )
        _gaussblur_transform = A.augmentations.transforms.GaussianBlur(
            blur_limit=(5, 11),
            sigma_limit=0,
            always_apply=False,
            p=0.1
        )
        _fancypca_transform = A.augmentations.transforms.FancyPCA(
            alpha=0.1,
            always_apply=False,
            p=0.3
        )
        _emboss_transform = A.augmentations.transforms.Emboss(
            alpha=(0.4, 0.5),
            strength=(0.4, 0.7),
            always_apply=False,
            p=0.2
        )
        _downscale_transform = A.augmentations.transforms.Downscale(
            scale_min=0.2,
            scale_max=0.9,
            interpolation=1,
            always_apply=False,
            p=0.1
        )
        _color_jitter_transform = A.augmentations.transforms.ColorJitter(
            brightness=0.4,
            contrast=0.3,
            saturation=0.4,
            hue=0.0,
            always_apply=False,
            p=0.3
        )
        _blur_transform = A.augmentations.transforms.Blur(
            blur_limit=10,
            always_apply=False,
            p=0.1
        )
        _clahe_transform = A.augmentations.transforms.CLAHE(
            clip_limit=2.0,
            tile_grid_size=(12, 12),
            always_apply=False,
            p=0.1
        )
        _resize_transform = A.augmentations.geometric.resize.Resize(
            height=size_,
            width=size_,
            interpolation=1,
            always_apply=True,
            p=1
        )
        _normalization = A.Normalize(mean=NORMALIZATION_PARAMS['color mean'], std=NORMALIZATION_PARAMS['color std'])
        train_transforms = A.Compose([
            _color_jitter_transform,
            _emboss_transform,
            _fancypca_transform,
            _opticaldistorsion_transform,
            _blur_transform,
            _clahe_transform,
            _isonoise_transorm,
            _motionblur_transorm,
            _glassblur_transform,
            _horizontalflip_transorm,
            _randomrotation_transform,
            _resize_transform,
            _downscale_transform,

            _normalization
        ])

        return train_transforms

    @staticmethod
    def validation_transform(size_: int):
        _normalization = A.Normalize(mean=NORMALIZATION_PARAMS['color mean'], std=NORMALIZATION_PARAMS['color std'])
        val_transform = A.Compose([
            A.augmentations.geometric.resize.Resize(
                height=size_,
                width=size_
            ),

            _normalization
        ])
        return val_transform


class AlbuTransformCollection(TransformCollection):
    @staticmethod
    def train_transform(size_: int):
        _normalization = A.Normalize(mean=NORMALIZATION_PARAMS['color mean'], std=NORMALIZATION_PARAMS['color std'])
        train_transform = A.Compose([
            A.augmentations.geometric.resize.Resize(
                height=size_,
                width=size_
            ),
            A.Rotate(
                limit=20,
                p=0.5
            ),
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.3,
                interpolation=1,
                border_mode=4,
                value=None,
                mask_value=None,
                always_apply=False,
                p=0.1
            ),
            A.HorizontalFlip(
                p=0.5
            ),
            A.ISONoise(
                color_shift=(0.01, 0.01),
                intensity=(0.01, 0.01),
                always_apply=False,
                p=0.5
            ),
            A.OpticalDistortion(
                distort_limit=0.3,
                shift_limit=0.1,
                interpolation=1,
                border_mode=4,
                value=None,
                mask_value=None,
                always_apply=False,
                p=0.5
            ),
            A.GaussNoise(
                var_limit=(20.0, 100.0),
                mean=0,
                per_channel=True,
                always_apply=False,
                p=0.3
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.1,
                brightness_by_max=False,
                always_apply=False,
                p=0.5
            ),

            _normalization
        ])
        return train_transform

    @staticmethod
    def validation_transform(size_: int):
        _normalization = A.Normalize(mean=NORMALIZATION_PARAMS['color mean'], std=NORMALIZATION_PARAMS['color std'])
        val_transform = A.Compose([
            A.augmentations.geometric.resize.Resize(
                height=size_,
                width=size_
            ),

            _normalization
        ])
        return val_transform