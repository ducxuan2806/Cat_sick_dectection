import os
import cv2
import albumentations as A

class Preprocessing:
    def __init__(self, input_image = "", input_label= "", output_image = "", output_label = ""):
        self.input_image = input_image
        self.input_label = input_label
        self.output_image = output_image
        self.output_label = output_label
        os.makedirs(self.output_image, exist_ok=True)
        os.makedirs(self.output_label, exist_ok=True)
        self.transform = self.preprocessing_pipeline()

    def denoise(self, img, ksize=(5, 5), **kwargs):
        return cv2.GaussianBlur(img, ksize, 0)

    def equalize_histogram(self, img, **kwargs):
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YCrCb2BGR)

    def preprocessing_pipeline(self, target_size=(640, 640), ):
        return A.Compose([
            # A.Lambda(image=self.denoise),
            # A.Lambda(image=self.equalize_histogram),
            A.Resize(*target_size),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    def preprocess_with_folder(self):
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
                            class_labels.append(int(class_id))

            # áp dụng pipeline
            try:
                transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
            except Exception as e:
                print(f"❌ Lỗi khi transform ảnh {filename}: {e}")
                continue

            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']
            transformed_labels = transformed['class_labels']

            # lưu ảnh
            out_img_path = os.path.join(self.output_image, filename)
            cv2.imwrite(out_img_path, transformed_image)

            # lưu nhãn
            out_label_path = os.path.join(self.output_label, os.path.splitext(filename)[0] + ".txt")
            with open(out_label_path, 'w') as f:
                for bbox, cls_id in zip(transformed_bboxes, transformed_labels):
                    f.write(f"{cls_id} {' '.join(f'{x:.6f}' for x in bbox)}\n")

            print(f"✅ Đã xử lý và lưu: {filename}")

