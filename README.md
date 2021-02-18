# Video Classification UCF101
Video Classification on UCF101 dataset


# Preprocessing

First, download the dataset from UCF Repository [https://www.crcv.ucf.edu/data/UCF101.php] then run the "preprocessing.py" file to preprocess UCF101 dataset.

In the preprocessing phase I used a different technique which extracted exactly 1,650 frames per category
meaning 1,650 x 101 = 1,66,650 frames or you can say images in whole dataset

You can download the dataset from my kaggle profile:
	[https://www.kaggle.com/oetjunhbxu/video-classification-ucf101]

# About Dataset

UCF101 folder: (where original dataset is located)

~100 to 150 videos in each category

~13,320 total videos in UCF101 folder

dataset folder: (where only one videos is there for each category after combining)

~only 1 video in each category

~101 total videos in dataset folder after combining all the videos

training folder: (where training_set is located)

~1500 frames in each category

~1500 x 101 = 151500 total frames

testing folder: (where testing_set is located)

~150 frames in each category

~150 x 101 = 15150 total frames


frames in training_set = 151500

frames validation set = 20% of training set (30300 frames)

frames in testing_set = 15150 (10% of training set)


# Model Analysis:
models used:

	ResNet50
	
	ResNet101

	ResNet152
	
	ResNet50V2
	
	ResNet101V2

	ResNet152V2
	
	MobileNet
	
	MobileNetV2

MobileNet and MobileNetV2 are worst model to perform Video Classification because they aren't made for heavy datasets infact they are made for mobile and embedded devices, hence named "mobile". Also MobileNets are giving good accuracies but have higher losses, that's why we discarded this model

ResNet50, ResNet101, ResNet152, ResNet50V2, ResNet101V2 and ResNet152V2 all of them are giving much impressive results than their counterparts (MobileNets) but took much time for training because of the fact that it contains more deeper and hidden layers.


# Required Parameters

Actual Dataset Path

dataset = "UCF-101/"

After Combining all videos of the Dataset, the recreated Dataset Path

dataset2 = "dataset/"

Training Path

train_path = "training_set/"

Testing path

test_path = "testing_set/"

Total number of frames to be extracted from a single category

no_of_frames = 1650

Number of epochs

epochs = 20

Batch size

batch_size = 32

Number of classes/categories (101)

n_classes = 101

Optimizer Used

optimizer = "Adam"

Loss Metric used for every model is one and same

loss_metric = "categorical_crossentropy"

Softmax function is used for last layer

last_layer_activation_function = "softmax"


input shape of ResNet50, ResNet101, ResNet152, ResNet50V2, ResNet101V2, ResNet152V2, MobileNet and MobileNetV2 are all the same and that is: (224, 224, 3) => [image height, image width and number of channels]
