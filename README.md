## Adaptive Order Selector and Extractor Graph Neural Network



## Environment Settings    
- pytorch 1.9.0
- numpy 1.21.6
- torch-geometric 2.2.0 
- tqdm 4.64.1
- scipy 1.7.3
- seaborn 0.12.2
- scikit-learn 1.0.2

## Experimental Setting
All the experiments are performed by running PyTorch on a GPU machine at Nvidia GeForce RTX 2080 Ti. For all datasets, we use the public data segmentation provided by Geom-gcn, averaging an average train/val/test split ratio of 60%/20%/20% nodes per class. For fairness, all models report means with standard deviations over 10 random splits. We use the Adam optimizer and 1000 epochs to train our model and baselines.

## Node classification on real-world datasets (./data)
We evaluate the performance of AOSENet against the competitors on 10 real-world datasets.

### Datasets
We provide the datasets in the folder './data' and you can run the code directly, or you can choose not to download the datasets('./data') here. The code will automatically build the datasets through the data loader of Pytorch Geometric.

### Running the code

You can run the following script directly and this script describes the hyperparameters settings of AOSENet on each dataset.
```sh
sh best.sh
```
or run the following Command 
+ Cora
```sh
python train.py   --dataset cora     --lr 0.08   --dropout 0.5  --weight_decay 5e-4 --l 3   --device 2  
```
+ Cornell
```sh
python train.py    --dataset cornell  --lr 0.07  --dropout 0.5  --weight_decay 5e-4  --l 5  --device 3  
```



## Contact
If you have any questions, please feel free to contact me with guoling@njust.edu.cn



