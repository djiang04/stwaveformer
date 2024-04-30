# stwaveformer

This is our final project for 10-708. The datasets are included in the folders above. If not, the datasets are available in the /data folders within each model folder.

Packages required:
```
pytorch>=1.11
numpy
pandas
matplotlib
pyyaml
pickle
torchinfo
json
pywt
```

To Run STAWaveformer:

```
cd STAformer/model
python train.py -d <dataset> -p DWT
```
To Run STAWave2former:

```
cd STA2former/model
python train.py -d <dataset> -p DWT
```
To Run STAPCAformer:

```
cd STAformer/model
python train.py -d <dataset> -p PCA
```
To Run STAPCA2former:

```
cd STA2former/model
python train.py -d <dataset> -p PCA
```
Datasets:
```
METRLA
PEMSBAY
PEMS03
PEMS04
PEMS07
PEMS08
```