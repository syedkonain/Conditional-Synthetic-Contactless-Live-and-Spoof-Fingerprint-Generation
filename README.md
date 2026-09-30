# Conditional-Synthetic-Contactless-Live-and-Spoof-Fingerprint-Generation

## DB2 Dataset

Download 1,200 synthetically generated live contactless fingerprints along with their corresponding synthetic contactless spoof fingerprints for 5 spoof materials across 8 finger classes (1–8). Each class contains 150 contactless fingerprints.

**Download link:**
https://drive.google.com/drive/folders/1w35ibk4ednIMqGRV-ci-pY4avrioHghV?usp=sharing

## DB3 Dataset

Download 1,200 synthetically generated live contactless fingerprints along with their corresponding synthetic contactless spoof fingerprints for 5 spoof materials across 8 finger classes (1–8). Each class contains 150 contactless fingerprints.

**Download link:**
https://drive.google.com/drive/folders/1W5rR4N505jppunWSl8vMIr6mg1O2Hl2v?usp=sharing

## Pretrained Model Weights

The trained weights for both models, BicycleGAN-based multiple impressions generator, and the weights corresponding to 5 spoof materials, are available at:

https://drive.google.com/drive/folders/1r10OJ6P4vLTdK-kUH38il5NrRDUdWsCX?usp=sharing

## Usage

### DB2 Dataset Generation

1-Clone the official StyleGAN2-ADA repository.

2-Install the required dependencies.

3-Download the official BicycleGAN and CycleGAN repositories inside the same StyleGAN2-ADA directory.

4-Install dependencies.

5-Download our pretrained weights for DB2 live, multiple_impressions, and all 5 spoof materials, and place them in the project folder.

6-Download the gen_db2_cl.py script for image generation.

#### Generation Parameters

| Argument | Description |
|--------|------------|
| `--class` | Select the desired finger class (1–8) |
| `--name` | Choose `live` or a spoof material |
| `--num-impressions` | Number of impressions per fingerprint |
| `--num-images` | Number of unique fingerprints to generate |
| `--seeds` | Specify seed range for controlled generation |

#### Spoof Material Names
- Live  
- EcoFlex  
- PlayDoh  
- Wood Glue   
- Latex  
- Photo Paper 

#### Example Command
```bash
python gen_db2_cl.py \
  --outdir DB2 \
  --network /home/DB2.pkl \
  --num-impressions 3 \
  --num-images 50 \
  --class 1 \
  --name live
```

### DB3 Dataset Generation

1-Clone the official StyleGAN3 repository.

2-Install the required dependencies.

3-Download the official BicycleGAN and CycleGAN repositories inside the same StyleGAN3 directory.

4-Install the dependencies.

5-Download our pretrained weights for DB3 live, multiple_impressions, and all 5 spoof materials, and place them in the project folder.

6-Download the gen_db3_cl.py script for image generation.

#### Generation Parameters

| Argument | Description |
|--------|------------|
| `--class` | Select the desired finger class (1–10) |
| `--name` | Choose `live` or a spoof material |
| `--num-impressions` | Number of impressions per fingerprint |
| `--num-images` | Number of unique fingerprints to generate |
| `--seeds` | Specify seed range for controlled generation |

#### Spoof Material Names
- Live  
- EcoFlex  
- PlayDoh  
- Wood Glue   
- Latex  
- Photo Paper 

#### Example Command
```bash
python gen_db3_cl.py \
  --outdir DB3 \
  --network /home/DB3.pkl \
  --num-impressions 3 \
  --num-images 50 \
  --class 1 \
  --name live
```


## Acknowledgments

We utilized the official implementations of StyleGAN2-ADA (https://github.com/NVlabs/stylegan2-ada-pytorch), StyleGAN3 (https://github.com/NVlabs/stylegan3), BicycleGAN (https://github.com/junyanz/BicycleGAN), and CycleGAN https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.
