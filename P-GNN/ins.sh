pip uninstall torch-scatter
pip uninstall torch-sparse
pip uninstall torch-cluster
pip uninstall torch-spline-conv
pip uninstall torch
pip install torch==1.8.0+cu101 torchvision==0.9.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html
pip install torch-geometric
unzip data/ppi.zip -d data/
pip install tensorboardx
