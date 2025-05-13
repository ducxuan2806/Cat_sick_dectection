import albumentations as A
import cv2
import os

class Augmentations:
    def __init__(self, input_image, input_label, output_image, output_label):
        self.input_image = input_image
        self.output_image = output_image
        self.input_label = input_label
        self.output_label = output_label

    def rotate(self, image, bboxes, class_labels):
        aug = A.Rotate(limit=(90, 90), p=1, border_mode=cv2.BORDER_CONSTANT)
        return aug(image=image, bboxes= bboxes, class_labels= class_labels)

    def shear(self, image, bboxes, class_labels):
        aug = A.Affine(shear={"x": (-20, 20)}, p=1)
        return aug(image=image, bboxes=bboxes, class_labels=class_labels)

    def crop_image(self, image, bboxes, class_labels):
        aug = A.RandomCrop(width=200, height=200, p=1)
        return aug(image=image, bboxes=bboxes, class_labels=class_labels)

    def adjust_brightness(self, image, bboxes, class_labels):
        aug = A.RandomBrightnessContrast(brightness_limit= 0.3, contrast_limit= 0.3, p = 1)
        return aug(image=image, bboxes=bboxes, class_labels=class_labels)


    def adjust_saturation(self, image, bboxes, class_labels):
        aug = A.HueSaturationValue(sat_shift_limit=30, p=1)
        return aug(image=image, bboxes=bboxes, class_labels=class_labels)

    def get_diverse_augmentation_pipeline(self):
        return A.Compose([
            A.OneOf([
                A.Rotate(limit=(90, 90), p=0.33),
                A.Rotate(limit=(180, 180), p=0.33),
                A.Rotate(limit=(270, 270), p=0.34)
            ], p=1.0),

            A.Affine(shear={"x": (-20, 20)}, p=0.8),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),

            A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=30, val_shift_limit=0, p=0.7),
        ],
            bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    def augmentation_with_folder(self, max_augmentation = 5):
        for filename in os.listdir(self.input_image):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(self.input_image, filename)
            label_path = os.path.join(self.input_label, os.path.splitext(filename)[0] + ".txt")

            # đọc ảnh
            image = cv2.imread(img_path)
            if image is None:
                print(f"⚠️ Không đọc được ảnh: {img_path}")
                continue

            bboxes = []
            class_labels = []
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id, x, y, bw, bh = parts
                            bboxes.append([float(x), float(y), float(bw), float(bh)])
                            class_labels.append(int(float((class_id))))

            base_name = os.path.splitext(filename)[0]
            # áp dụng pipeline
            for i in range(max_augmentation):
                try:
                    transform = self.get_diverse_augmentation_pipeline()
                    augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)

                    aug_img = augmented['image']
                    aug_bboxes = augmented['bboxes']
                    aug_labels = augmented['class_labels']

                    # Lưu ảnh
                    out_img_name = f"{base_name}_aug_{i}.jpg"
                    out_img_path = os.path.join(self.output_image, out_img_name)
                    cv2.imwrite(out_img_path, aug_img)

                    # Lưu label
                    out_label_name = f"{base_name}_aug_{i}.txt"
                    out_label_path = os.path.join(self.output_label, out_label_name)
                    with open(out_label_path, 'w') as f:
                        for bbox, cls in zip(aug_bboxes, aug_labels):
                            f.write(f"{cls} {' '.join(map(str, bbox))}\n")

                except Exception as e:
                    print(f"❌ Lỗi khi augment ảnh {filename} (aug {i}): {e}")









