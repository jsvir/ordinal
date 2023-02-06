
[Deep Ordinal Regression using Optimal Transport Loss and Unimodal Output Probabilities](https://arxiv.org/abs/2011.07607)


### Installation:
1. (Verify that you have Python >= 3.8)
2. clone the repo to your local work directory: `git clone https://github.com/jsvir/ordinal.git`
3. go to the dir: `cd ordinal`
4. install the requirements: `pip install -r requirements.txt` (some requirements are not strict)

### Supported datasets:
1. [Adience](https://talhassner.github.io/home/projects/Adience/Adience-data.html)
2. [Historical Color](http://graphics.cs.cmu.edu/projects/historicalColor/)
3. [FG-NET](https://yanweifu.github.io/FG_NET_data/)
4. [RetinaMNIST](https://medmnist.com/)

### Data preparation

1. DR:
   * install submodule project: `git submodule init && git submodule update --remote`
   * modify dr config with your parameters

2. Adience:
   * download the dataset manually from [here](https://talhassner.github.io/home/projects/Adience/Adience-data.html)
   * update `conf/adience.yaml` with `data_images=path_to_aligned_you_downloaded`
   * the data split is the same as provided [here](https://github.com/GilLevi/AgeGenderDeepLearning/tree/master/Folds/train_val_txt_files_per_fold) and is already exists in this project `dataset/adience_folds`

3. HCI:
   * download the dataset images from [here](http://graphics.cs.cmu.edu/projects/historicalColor/)
   * update `conf/hci.yaml` with `data_root=you_path_to_HistoricalColor-ECCV2012/data/imgs/decade_database`

### How to run

Run experiments using the script `python run_exps.py`


### Adding new datasets:
1. Update the `parts/dataset/dataset.py` with your new dataset (use the abstact class `parts/dataset/dataset.OrdinalDataset` as a template)
2. Add a new ptl module in `parts/ptl.py` (with your custome augmentations, data split, etc.)
3. Add new config in `conf`

### Adding new models:
1. It's important to distinguish between pytorch-lightning **module** and torch.nn.Module **model**
2. Each new module must inherit from `parts.ptl.ptl_modules` and `parts.models.models_list` where the first one is custom to a dataset and the last one implements some methods proposed by different researchers.
3. If you have any questions regarding new models please open an issue in this project.

