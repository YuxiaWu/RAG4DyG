[SIGIR 2025] Code and data for "Retrieval Augmented Generation for Dynamic Graph Modeling"

# Requirements and Installation

python>=3.9

pytorch>=1.9.1

transformers>=4.24.0

torch_geometric>=1.7.2

# Datasets and Preprocessing

## Raw data:

- UCI and  ML-10M: the raw data is the same with  https://github.com/aravindsankar28/DySAT

- Hepth: The dataset can be downloaded from the KDD cup:  https://www.cs.cornell.edu/projects/kddcup/datasets.html

- MMConv: we provide the raw data downloaded from https://github.com/liziliao/MMConv. It is a text-based multi-turn dialog dataset. We preprocess the data by representing the dialog as a graph for each turn based on the annotated attributes. We provide the preprocessed data in `all/data/dialog`

- Wikipedia: The dataset can be downloaded from https://object-arbutus.cloud.computecanada.ca/tgb/tgbl-wiki-v2.zip. It is one of the dataset of TGB (https://github.com/shenyangHuang/TGB)
  
- Enron: The dataset can be downloaded from https://drive.google.com/drive/folders/1QFxHIjusLOFma30gF59_hcB19Ix3QZtk. It is one of the dataset of DTB (https://github.com/zjs123/DTGB)
  
- Reddit: The dataset can be downloaded from https://snap.stanford.edu/data/soc-RedditHyperlinks.html

## Let's do preprocessing!  

All the datasets and preprocessing code are in folder `/all_data`. For each dataset, run:

`python preprocess.py ` 


The preprocessed data contains:

- `ml_dataname.csv`: the columns: *u*, *i* is the node Id. *ts* is the time point. *timestamp* is the coarse-grained time steps for temporal alignment.
- `ml_dataname.npy`: the raw link feature. 
- `ml_dataname_node.npy`: the raw node feature. 

Transfer the preprocessed data into sequences for the Transformer model: 

`bash csv2res.sh`

The final data is saved in  `./resources`, including the train/val/test data.


The data files for each dataset:

- train.link_prediction: The sequences of the training set. This will be used as the retrival pool.
- test.link_prediction: The sequences of the test set. 
- test_gt.link_prediction: The ground truth of the test set. 
- val.link_prediction: The sequences of the validation set.
- val_gt.link_prediction: The ground truth of the validation set.

# 1. Train sequence model using SimpleDyG

## 1. Run Backbone: SimpleDyG

```python
bash scripts/SimpleDyG/train_datasetname.sh
```

The output checkpoints will be saved at `simpledyg_ckpt/datasetname/timestep/seed/gpt2`, which are used for the retriever and generator.

# 2. Retrieval Pool Annotation

Compute the Jaccard similarity of `y` and rank the similarity of each query with the samples in the retrieval pool.

We set the treshold to 0.8, which means that the samples with Jaccard similarity greater than 0.8 are considered as positive samples.

Run the following codes to generate the retrieval pool. Get the train/val/test data files to train the retriever. Get the ground truth topK demonstrations for each sequence in the retrieval pool to train the generator.

We use wandb to record the training process for retriever and generator. Put the key in the `wandb.login(key='your_key')` in the `main_retrieval.py`,`main_generator`, `train/train_retriever.py`, `train/train_generator.py`. If you don't want to use wandb, you can comment the wandb code in the `main_retrieval.py` and `train/train_retriever.py`

```python
    proj_name = 'RAG4DyG_retriever/generator'+args.dataset
    wandb.login(key='your key')
    wandb.init(project=proj_name, name=args.run_name, save_code=True, config=args)
    wandb.run.log_code(".")
```

```python
python retrieval_data_annotation.py UCI_13 12 0.8

python retrieval_data_annotation.py hepth 11 0.8

python retrieval_data_annotation.py dialog 15 0.8

python retrieval_data_annotation.py wikiv2 15 0.8

python retrieval_data_annotation.py enron 16 0.8

python retrieval_data_annotation.py reddit 11 0.8
```

Output:

You will get:

(1) The files for retriever saved in: `save_path = os.path.join('./resources/', dataset, str(timestamp), 'train_retrieval')`

saved files: index and score for train/val/test:

- 'train_index.retrieval', 'train_score.retrieval': Note that train_index is the sample pair index (e.g. 0 34 174, which means the 0th sample is the query and the 34th sample in the training set is the positive sample, 174 is the negative sample)

- 'test_index.retrieval', 'test_score.retrieval': the index is the ranked topk index of the retrieval pool for each query in the test set. the score is the similarity score of all the candidates in the retrieval pool.

- 'val_index.retrieval', 'val_score.retrieval': same meaning with test_index.retrieval and test_score.retrieval

(2) The files for generator saved in: `save_path_gen = os.path.join('./resources/train_generator', dataset, str(timestamp), "train_gt_topk")`

# 3. Train Retriever

## 3.1 Preprocess

Get the query time for each sequence in the retrieval pool for time-aware contrastive learning.

Input: 

the dataset name, timestep, the `ml_dataset_name.csv` file with the interction time for each edge.

```python
     python get_train_query_time.py 'UCI_13' '12' 
     python get_train_query_time.py 'hepth' '11' 
     python get_train_query_time.py 'dialog' '15' 
     python get_train_query_time.py 'wikiv2' '15' 
     python get_train_query_time.py 'enron' '16'
     python get_train_query_time.py 'reddit' '11'

```

For each dataset, the time scale (time granularity) is different. The time scale is defined in the `get_train_query_time.py` as follows:

```python    
    scales = {
        'UCI_13': 3600*24,
        'hepth': 3600*24*30,
        'dialog': 1,
        'wikiv2': 3600*24,
        'enron': 1,
        'reddit': 1
    }
```

Output: 

`os.path.join('resources', dataset_name + '_train_query_time.pt')`

Then the query_time file is used for calculating the time differences between two sequences.

## 3.2 Training

main_retrieval.py

Run the following codes to train the retriever:

```python
bash scripts/train_retriever/train_retriever_datasetname.sh
```

The retrieval results for test and validation datasets are saved in `resources/retrieval_result/{args.dataset}/`:

- 'test_index.gen' and 'test_score.gen': The ranked index based on the similarity score of the retrieval pool for each query in the test set.

- 'val_index.gen' and 'val_score.gen': same meaning with test_index.gen and test_score.gen

## 3.3 Evaluation: 

```python
bash scripts/train_retriever/eval_retriever_datasetname.sh
```


# 4. Train Generator

Remember to put your wandb key in the `wandb.login(key='your_key')` in the `main_generator.py`. If you don't want to use wandb, you can comment the wandb code in the `main_generator.py` and `train/train_generator.py`

## 4.1 Training

```python
bash scripts/train_generator/train_rag_graphpooling_datasetname_seed.sh
```

## 4.2 Evaluation: 

```python
bash scripts/train_generator/eval_rag_graphpooling_datasetname_seed.sh
```

# Citation

```
@article{wu2024retrieval,
  title={Retrieval Augmented Generation for Dynamic Graph Modeling},
  author={Wu, Yuxia and Fang, Yuan and Liao, Lizi},
  journal={arXiv preprint arXiv:2408.14523},
  year={2024}
}

```
