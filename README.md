# Two_Streams_Project

#Package dependencies
```
Python 3.5
Pytorch
OpenCV
Tochvision
OpenCV-contrib-python
```

# Commands
To convert dataset videos into frames:
```
python dataset_video_converter.py --src_dir ./HAA500/videos --store_dir ./two-stream-pytorch-master/datasets/HAA500_frames
```
To train the TinyMotionNet (in MotionNet folder): 
```
python flownet_train.py
```
To train the temporal stream (in two-stream-pytorch-master folder):
```
python temporal_net_training.py -m flow --new_length=10 --epochs 350 --b 32 --lr 0.001 --lr_steps 200 300
```
If you want to continue training a previously trained temporal stream model (in two-stream-pytorch-master folder):
```
python temporal_net_training.py -m flow --new_length=10 --epochs 350 --b 32 --lr 0.001 --lr_steps 200 300 -- resume ./checkpoints/"model_name"
```
If you want to evaluate a trained temporal stream model (in two-stream-pytorch-master folder):
```
python temporal_net_training.py -m flow --new_length=10 --epochs 350 --b 32 --lr 0.001 --lr_steps 200 300 --resume ./checkpoints/"model_name" -e "number of epochs to evaluate on (minimum 1)"
```
To train the spatial stream (in two-stream-pytorch-master folder): 
```
python spatial_stream_gpu.py ./datasets/HAA500_frames/
```
If you want to continue training a previously trained spatial stream model (in two-stream-pytorch-master folder):
```
python spatial_stream_gpu.py ./datasets/HAA500_frames/ -- resume ./checkpoints/"model_name"
```
If you want to evaluate a trained spatial stream model (in two-stream-pytorch-master folder):
```
python spatial_stream_gpu.py ./datasets/HAA500_frames/ --resume ./checkpoints/"model_name" -e "number of epochs to evaluate on (minimum 1)"
```

# Contributions
```
Nick wrote:
dataset_video_converter.py
spatial_stream_gpu.py: modified original code from paper to add functionality, allow for resuming training model, modified implementation to pytorch 1.10.1, and work with HAA500 dataset
```
```
Jeff wrote:
The code in MotionNet folder (joint work with stefano)
temporal_net_training.py: modified original code from paper to add functionality, allow for resuming training model, modified implementation to pytorch 1.10.1, and work with HAA500 dataset
```
