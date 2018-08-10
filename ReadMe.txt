# I am using TensorFlow to do image classification
# here is the Blog and which i learned https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0

# please follow the instruction to install tensorFlow python
# Extract the SRC Zip file
# since i am using MobileNet CNN which works on 244x244 image, so resize(scaling, re-sampling) the image to achieve 244x244. it will aspect ratio so that the images wont stretch.
# This code is to resize the image and save it in tensorFlow Directory arrange it in folder vise. species name is the folder name

# python -m scripts.load_image --train_folder=C:/my/test/train/ --train_file=C:/my/test/train.csv

# Then for training following command

# for windows

set IMAGE_SIZE=224
set ARCHITECTURE=mobilenet_1.0_%IMAGE_SIZE%

# CNN  i am using is Mobilenet_1.0_224

python -m scripts.retrain --bottleneck_dir=tf_files/bottlenecks --model_dir=tf_files/models/ --summaries_dir=tf_files/training_summaries/"%ARCHITECTURE%" --output_graph=tf_files/retrained_graph.pb --output_labels=tf_files/retrained_labels.txt  --architecture="%ARCHITECTURE%" --how_many_training_steps 500 --image_dir=tf_files/animals

# "how_many_training_steps 500" in 500 steps we will get 95% accuracy, for more accuracy you can increase steps up to 4000(max limit)



# This is for testing.
# Following code will resize the test image to provide a proper analyses and accuracy

python -m scripts.load_test_image --test_folder=C:/my/test/test/ --test_file=C:/my/test/test.csv

Finally run this code to get the submissions. check the result.csv in meta-data folder inside tensorflow

# python -m scripts.test_submission --test_file=C:/my/test/test.csv
