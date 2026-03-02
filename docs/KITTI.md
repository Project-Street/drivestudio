# Preparing KITTI Dataset
## 1. Download Dataset

To obtain the KITTI dataset:

1. Visit the [KITTI Raw Data](https://www.cvlibs.net/datasets/kitti/raw_data.php) official website.
2. Register for an account to access the download links.
3. Choose the specific dates and drives you need.
4. Download the following components:
   - Synchronized and unrectified data
   - Calibration files
   - Tracklets
5. After downloading, organize the files according to the file structure shown below.

##### File Structure of Raw Data
```
Project_path/Kitti/raw/
├── 2011_09_26
│   ├── 2011_09_26_drive_0001_sync
│   │   ├── image_00
│   │   ├── image_01
│   │   ├── image_02
│   │   ├── image_03
│   │   ├── oxts
│   │   ├── velodyne_points
│   │   └── tracklet_labels.xml
│   ├── 2011_09_26_drive_0002_sync
│   │   └── ... (similar structure as 0001_sync)
│   ├── ...
│   ├── calib_cam_to_cam.txt
│   ├── calib_imu_to_velo.txt
│   └── calib_velo_to_cam.txt
├── 2011_09_28
│   ├── 2011_09_28_drive_0001_sync
│   │   └── ... (similar structure as 0001_sync)
│   ├── ...
│   ├── calib_cam_to_cam.txt
│   ├── calib_imu_to_velo.txt
│   └── calib_velo_to_cam.txt
└── ...
```

## 2. Install the Development Toolkit
``` shell
pip install pykitti
```

## 3. Process Raw Data

To process the raw KITTI data, use the following command:

``` shell
# export PYTHONPATH=\path\to\project
python datasets/preprocess.py \
    --data_root data/kitti/raw \
    --dataset kitti \
    --split 2011_09_26 \
    --split_file data/kitti_example_scenes.txt \
    --target_dir data/kitti/processed \
    --workers 32 \
    --process_keys images lidar pose calib dynamic_masks objects
```

The extracted data will be stored in the `data/kitti/processed` directory.

## 4. Extract Masks

To generate:

- **sky masks (required)** 
- fine dynamic masks (optional)

Follow these steps:

#### Run Mask Extraction Script

```shell
python datasets/tools/extract_masks.py \
    --data_root data/kitti/processed \
    --split_file data/kitti_example_scenes.txt \
    --process_dynamic_mask
```

Note: The `--process_dynamic_mask` flag is included to process fine dynamic masks along with sky masks.

This process will extract the required masks from your processed data.

## 5. Human Body Pose Processing

#### Prerequisites
To utilize the SMPL-Gaussian to model pedestrians, please first download the SMPL models.

1. Download SMPL v1.1 (`SMPL_python_v.1.1.0.zip`) from the [SMPL official website](https://smpl.is.tue.mpg.de/download.php)
2. Move `SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl` to `PROJECT_ROOT/smpl_models/SMPL_NEUTRAL.pkl`

SMPL-Nodes (SMPL-Gaussian Representation) requires Human Body Pose Sequences of pedestrians. We've developed a human body pose processing pipeline for in-the-wild driving data to generate this information. There are two ways to obtain these data:

#### Option 1: Download Preprocessed Human Pose Data

We have uploaded preprocessed human pose data for a subset of KITTI scenes to [Google Drive](https://drive.google.com/drive/folders/187w1rwEZ5i9tb4y-dOJVTnIZAtKPR7_j). You can download and unzip these files without installing any additional environment.

```shell
# https://drive.google.com/file/d/1eAMNi5NFMU8T7tjQBT_jzxeX-yJRwVKM/view?usp=drive_link
# filename: kitti_preprocess_humanpose.zip
cd data
gdown 1eAMNi5NFMU8T7tjQBT_jzxeX-yJRwVKM

unzip kitti_preprocess_humanpose.zip
rm kitti_preprocess_humanpose.zip
```

#### Option 2: Run the Extraction Pipeline

To process human body poses for other KITTI scenes or to run the processing pipeline yourself, follow the instructions in our [Human Pose Processing Guide](./HumanPose.md).

## 6. Data Structure

After completing all preprocessing steps, the project files should be organized according to the following structure:

```shell
ProjectPath/data/
  └── kitti/
    ├── raw/
    │    ├── 2011_09_26/
    │    │   ├── 2011_09_26_drive_0001_sync/
    │    │   │   ├── image_00/
    │    │   │   ├── image_01/
    │    │   │   ├── image_02/
    │    │   │   ├── image_03/
    │    │   │   ├── oxts/
    │    │   │   ├── velodyne_points/
    │    │   │   └── tracklet_labels.xml
    │    │   ├── ...
    │    │   ├── calib_cam_to_cam.txt
    │    │   ├── calib_imu_to_velo.txt
    │    │   └── calib_velo_to_cam.txt
    │    └── ...
    └── processed/
         ├── 2011_09_26_drive_0001_sync/
         │  ├──images/             # Images: {timestep:03d}_{cam_id}.jpg
         │  ├──lidar/              # LiDAR data: {timestep:03d}.bin
         │  ├──ego_pose/           # Ego vehicle poses: {timestep:03d}.txt
         │  ├──extrinsics/         # Camera extrinsics: {cam_id}.txt
         │  ├──intrinsics/         # Camera intrinsics: {cam_id}.txt
         │  ├──sky_masks/          # Sky masks: {timestep:03d}_{cam_id}.png
         │  ├──dynamic_masks/      # Dynamic masks: {timestep:03d}_{cam_id}.png
         │  ├──fine_dynamic_masks/ # (Optional) Fine dynamic masks: {timestep:03d}_{cam_id}.png
         │  ├──objects/            # Object information
         │  └──humanpose/          # Preprocessed human body pose: smpl.pkl
         ├── 2011_09_26_drive_0002_sync/
         └── ...
```
