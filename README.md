# vid2depth_tf2

**Original Work**:
[vid2depth](https://github.com/tensorflow/models/tree/master/research/vid2depth)

```yaml
kitti-raw-uncompressed: dataset storage
data: dataset generation
checkpoints: training checkpoint files
inference: inference result storage
```

## 1. Installation

### (1) Python Packages

```shell
mkvirtualenv venv  # Optionally create a virtual environment.
pip install absl-py
pip install matplotlib
pip install numpy
pip install scipy
pip install tensorflow
```

### (2) For building the ICP op (work in progress)

* Bazel: https://bazel.build/

## 2. Datasets

### Download KITTI dataset (174GB)

```shell
mkdir -p ~/vid2depth/kitti-raw-uncompressed
cd ~/vid2depth/kitti-raw-uncompressed
wget https://github.com/mrharicot/monodepth/blob/master/utils/kitti_archives_to_download.txt
wget -i kitti_archives_to_download.txt
unzip "*.zip"
```

### Download Cityscapes dataset (110GB) (optional)

You will need to register in order to download the data.  Download the following files:

* leftImg8bit_sequence_trainvaltest.zip
* camera_trainvaltest.zip

### Download Bike dataset (17GB) (optional)

```shell
mkdir -p ~/vid2depth/bike-uncompressed
cd ~/vid2depth/bike-uncompressed
wget https://storage.googleapis.com/brain-robotics-data/bike/BikeVideoDataset.tar
tar xvf BikeVideoDataset.tar
```

## 3. Inference

### Download trained model

```shell
mkdir -p ~/vid2depth/trained-model
cd ~/vid2depth/trained-model
wget https://storage.cloud.google.com/vid2depth/model/model-119496.zip
unzip model-119496.zip
```

### Run inference

```shell
cd tensorflow/models/research/vid2depth
python inference.py \
  --kitti_dir ~/vid2depth/kitti-raw-uncompressed \
  --output_dir ~/vid2depth/inference \
  --video 2011_09_26/2011_09_26_drive_0009_sync \
  --model_ckpt ~/vid2depth/trained-model/model-119496
```

## 4. Training

### Prepare KITTI training sequences

```shell
# Prepare training sequences.
cd tensorflow/models/research/vid2depth
python dataset/gen_data.py \
  --dataset_name kitti_raw_eigen \
  --dataset_dir ~/vid2depth/kitti-raw-uncompressed \
  --data_dir ~/vid2depth/data/kitti_raw_eigen \
  --seq_length 3
```

### Prepare Cityscapes training sequences (optional)

```shell
# Prepare training sequences.
cd tensorflow/models/research/vid2depth
python dataset/gen_data.py \
  --dataset_name cityscapes \
  --dataset_dir ~/vid2depth/cityscapes-uncompressed \
  --data_dir ~/vid2depth/data/cityscapes \
  --seq_length 3
```

### Prepare Bike training sequences (optional)

```shell
# Prepare training sequences.
cd tensorflow/models/research/vid2depth
python dataset/gen_data.py \
  --dataset_name bike \
  --dataset_dir ~/vid2depth/bike-uncompressed \
  --data_dir ~/vid2depth/data/bike \
  --seq_length 3
```

### Compile the ICP op (work in progress)

The ICP op depends on multiple software packages (TensorFlow, Point Cloud
Library, FLANN, Boost, HDF5).  The Bazel build system requires individual BUILD
files for each of these packages.  We have included a partial implementation of
these BUILD files inside the third_party directory.  But they are not ready for
compiling the op.  If you manage to build the op, please let us know so we can
include your contribution.

```shell
cd tensorflow/models/research/vid2depth
bazel build ops:pcl_demo  # Build test program using PCL only.
bazel build ops:icp_op.so
```

For the time being, it is possible to run inference on the pre-trained model and
run training without the icp loss.

### Run training

```shell
# Train
cd tensorflow/models/research/vid2depth
python train.py \
  --data_dir ~/vid2depth/data/kitti_raw_eigen \
  --seq_length 3 \
  --reconstr_weight 0.85 \
  --smooth_weight 0.05 \
  --ssim_weight 0.15 \
  --icp_weight 0 \
  --checkpoint_dir ~/vid2depth/checkpoints
```
