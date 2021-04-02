# Benchmarking_PGNN
This repository contains the source code to carry out benchmarking of position aware graph neural networks. In addition to what the standard repo already offers, we also provide the option for various aggregators. 
To run the code : 
```
cd P-GNN
./ins.sh
python main.py --model PGNN --layer_num 32 --anchor_num 32 --dropout_no --dataset ppi --hidden_dimm 2 --agg max
pyton node_classification.py --layer_num 32 --anchor_num 32 --dropout_no --dataset ppi --hidden_dimm 2 --agg mean
```
For further options on this, look in [args.py](P-GNN/args.py) . We also provide options to generate embeddings from the code and visualise them. For that, run
```
python main_graphic.py --model PGNN --dataset grid
```
You can find our results and tensorboard summaries at this [link](https://drive.google.com/drive/folders/1-DgRAV-yQ2zG_MbhnBpNPOOyc50LrK0n?usp=sharing)
