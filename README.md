WEED DETECTİON PROJECT

About the Project
This project was developed using the YOLOv8 model to detect weeds (Portulaca oleracea and Chenopodium album) in field images. It aims to contribute to the effective identification and management of weeds in agricultural areas.

FEATURES

Model: Trained using YOLOv8.
Target Objects: Portulaca oleracea and Chenopodium album species.
Dataset: Labeled images downloaded from Roboflow were used, and additional data were added to create a new dataset. The labels were also adjusted accordingly.
Data Augmentation: Various data augmentation techniques were applied to increase the number of images from 526 to 2800. After the data augmentation process, train_augment folders are automatically generated.
Training Files: The best results of the model training are stored in best.onnx and best.pt files. You can replace these files with your own training results.

SATASETS USED

Normal Final Dataset https://universe.roboflow.com/2-xx7t0/normal_final/dataset/1

Weed Detection Dataset https://universe.roboflow.com/dronepag/weed_detection-rrlf8-xxodx/dataset/1

Chenopodium Album Dataset https://universe.roboflow.com/university-of-agriculture-faisalabad-fqroe/chenopodium-album-vthkc/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true

