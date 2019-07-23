#!/bin/sh

################ SFOP ##################
printf "\n \n \n \n______________Setting up sfop detector_____________\n \n"

cd sfop
mkdir -p build
cd build
cmake ..
make
cd ../..

echo "sfop installation successful"
# if there are comments in the second line of the image, then remove them
# because c_img library cannot read these images 
bash sfop/pre_process.sh

############### SUPERPOINT MODEL #######################
printf "\n \n \n \n ____________Downloading pretrained model for SuperPointNetwork____________ \n \n"
cd superpoint
wget https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork/raw/master/superpoint_v1.pth
echo "SuperpointNet: Download complete"  
cd ..

################# D2NET MODEL ######################
printf "\n \n \n \n __________________Downloading pretrained model for D2-net_________________ \n \n"
cd d2net
wget https://dsmn.ml/files/d2-net/d2_tf.pth
cd ..


################# LIFT MODEL ######################
printf "\n \n \n \n __________________Downloading pretrained model for LIFT_________________ \n \n"
cd lift
mkdir models
cd models
mkdir base configs picc-best
cd base
wget https://github.com/cvlab-epfl/LIFT/raw/master/models/base/new-CNN3-picc-iter-56k.h5 
cd ../configs
wget https://github.com/cvlab-epfl/LIFT/raw/master/models/configs/picc-finetune-nopair.config 
cd ../picc-best
wget https://github.com/cvlab-epfl/LIFT/raw/master/models/picc-best/mean_std.h5
wget https://github.com/cvlab-epfl/LIFT/raw/master/models/picc-best/model.h5 
echo "LIFT: Download complete"
cd ../../../



echo "===> SETUP Complete"
echo "."
echo "."
echo "."
printf "\n \n \n=============================================== \n"
echo "Next: Extract keypoints and descriptors"
echo "Go through README and extract_features.py"
