from Augmentation import Augmentations

project_path = "C:/users/xduc2/PycharmProjects/Graduation/"
train_img = project_path + "Data/train/images"
train_label = project_path + "Data/train/labels"
preprocessed_train_img = project_path + "Data_Preprocessed/train/images"
preprocessed_train_label =  project_path + "Data_Preprocessed/train/labels"
augment_train_img = project_path + "Data_Augmented/train/images"
augment_train_label = project_path + "Data_Augmented/train/labels"


valid_img = project_path + "Data/valid/images"
valid_label = project_path + "Data/valid/labels"
preprocessed_valid_img = project_path + "Data_Preprocessed/valid/images"
preprocessed_valid_label = project_path + "Data_Preprocessed/valid/labels"
augment_valid_img = project_path + "Data_Augmented/valid/images"
augment_valid_label = project_path + "Data_Augmented/valid/labels"

test_img = project_path + "Data/test/images"
test_label = project_path + "Data/test/labels"
preprocessed_test_img = project_path + "Data_Preprocessed/test/images"
preprocessed_test_label = project_path + "Data_Preprocessed/test/labels"
augment_test_img = project_path + "Data_Augmented/test/images"
augment_test_label = project_path + "Data_Augmented/test/labels"
# Create dire

#Preprocessing

# train_preprocessing = Preprocessing(input_image= train_img, input_label= train_label, output_image= preprocessed_train_img, output_label= preprocessed_train_label)
# valid_preprocessing = Preprocessing(input_image= valid_img, input_label= valid_label, output_image= preprocessed_valid_img, output_label= preprocessed_valid_label)
#
# train_preprocessing.preprocess_with_folder()
# valid_preprocessing.preprocess_with_folder()

#Augmentation

train_augmentation = Augmentations(input_image= train_img, input_label= train_label, output_image= augment_train_img, output_label= augment_train_label)
valid_augmentation = Augmentations(input_image= valid_img, input_label= valid_label, output_image= augment_valid_img, output_label= augment_valid_label)
# train_augmentation.augmentation_with_folder(max_augmentation= 5)
valid_augmentation.augmentation_with_folder(max_augmentation= 5)

# test_augmentation = Augmentations(input_image= test_img, input_label= test_label, output_image= augment_test_img, output_label= augment_test_label)
# test_augmentation.augmentation_with_folder(max_augmentation= 5)
