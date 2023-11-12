# JUAL
### Joint Image Upsampling with Affinity Learning

## Get Started
### Prepare Environment[python>=3.7]

#### Hardware
- CPU Intel i5-12400F
- GPU NVIDIA RTX 3090
- RAM 16G*2 3200MHz

#### Installation
1. Clone repo

```sh
git clone https://github.com/LiQingCode/JUAL.git
cd JUAL
```

2. Install dependent packages

```sh
pip install -r requirements.txt
```

#### Dataset
1. NYUv2
2. Sintel
3. Middlebury
4. Lu
5. MIT-Adobe FiveK

You can directly download these datasets(Except for the MIT-Adobe FiveK):[Google Drive](https://drive.google.com/drive/folders/1EwbsIBJ5euKjD21yMRBvYvktHCjPnxBG?usp=sharing).

[MIT-Adobe FiveK (≈50GB)](https://data.csail.mit.edu/graphics/fivek/fivek_dataset.tar)

#### Trained Models
You can directly download these Models:[Google Drive](https://drive.google.com/drive/folders/1EwbsIBJ5euKjD21yMRBvYvktHCjPnxBG?usp=sharing).

#### Train

```sh
./depth_map_upsampling/train.py
./others_upsampling/train_hr.py

python train.py
python train_hr.py
```

#### Test

```sh
./depth_map_upsampling/test.py
./others_upsampling/test_hr.py

python test.py
python test_hr.py
```

#### Citiation
