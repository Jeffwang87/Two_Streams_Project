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
To train the spatial stream: 
```
python spatial_stream_gpu.py ./datasets/HAA500_frames/
```
If you want to continue training a previously trained spatial stream model:
```
python spatial_stream_gpu.py ./datasets/HAA500_frames/ -- resume ./checkpoints/"model_name"
```
If you want to evaluate a trained spatial stream model:
```
python spatial_stream_gpu.py ./datasets/HAA500_frames/ --resume ./checkpoints/"model_name" -e "number of epochs to evaluate on (minimum 1)"
```

# Contributions
```
Nick wrote:
dataset_video_converter.py
spatial_stream_gpu.py: modified original code from paper to add functionality, allow for resuming training model, modified implementation to pytorch 1.10.1, and work with HAA500 dataset
```
