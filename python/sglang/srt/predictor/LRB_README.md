# cache-coliseum

A benchmark for evaluating cache algorithms and predictor training, featuring Parrot and LightGBM implementations.

## Trace

Based on [ML caching with guarantees](https://github.com/chledowski/ml_caching_with_guarantees)<sup>[1]</sup>, we have reorganized the Brightkite and Cite datasets to conform to a unified format.

You can easily download all trace files, boost traces pickle and GBM Model Checkpoints from the Releases page.

## Environment

We use an environment based on Python 3.10.16 with Anaconda3 (Anaconda3-2021.05-Linux-x86_64.sh), we strongly recommend using Anaconda for Python package management and virtual environment.

You can also install your own environment based on another Python Version but only pay attention that:
- If you want to use the `Parrot` Model, you should install your torch env(in our benchmark, we use CUDA 12.4 to enable Torch GPU)
- Because of different Python pickle rules, the released `boost-traces.zip` may not work in your env, so you need to generate your own boost trace file.

## Usage

### Benchmark

A Simple usage:
```python
python -m benchmark --dataset xalanc --real --pred pleco --boost --boost_fr 
```

You will see the benchmark results of all supported algorithms on the console.

#### All Usages
```python
python -m benchmark [--dataset DATASET] [--test_all] [--device DEVICE] (--oracle | --real)
                   [--pred {parrot,pleco,popu,pleco-bin,gbm,oracle_bin,oracle_dis} [{parrot,pleco,popu,pleco-bin,gbm,oracle_bin,oracle_dis} ...]]
                   [--noise_type {dis,bin,logdis}] [--dump_file] [--output_root_dir OUTPUT_ROOT_DIR] [--verbose]
                   [--boost] [--num_workers NUM_WORKERS] [--boost_fr] [--boost_preds_dir BOOST_PREDS_DIR]
                   [--model_fraction MODEL_FRACTION] [--checkpoints_root_dir CHECKPOINTS_ROOT_DIR]
                   [--parrot_config_path PARROT_CONFIG_PATH] [--lightgbm_config_path LIGHTGBM_CONFIG_PATH]
```

- Dataset

  `--dataset`: Benchmark source dataset name, supported name:
    + `brightkite`
    + `citi`
    + `astar`
    + `bwaves`
    + `bzip`
    + `cactusadm`
    + `gems`
    + `lbm`
    + `leslie3d`
    + `libq`
    + `mcf`
    + `milc`
    + `omnetpp`
    + `sphinx3`
    + `xalanc`
    
    You can also support your own dataset.

  `--test_all`: This option only works for `brightkite` and `citi` datasets, meaning the entire traces will be used in these datasets. It cannot be used with `parrot` or `gbm` predictors as they require a training set.
- Device

  `--device`: Model target device, like `cpu`, `cuda:0` and `cuda:1`...

  This option only works for the `parrot` predictor, in other cases we only use CPU.
  
- Mode

  `real` and `oracle`: You must specify one of them in command.

  - Real Mode

    Supported Predictors:

    + `parrot`
    + `pleco`
    + `popu`
    + `pleco-bin`
    + `gbm`

    `verbose` default to True

    `boost` will use pickle traces (`num_workers` doesn't work)

    `noise_type` doesn't work
    
  - Oracle Mode

    Supported Predictors:

    + `oracle_dis`
    + `oracle_bin`
 
    `verbose` default to False

    `boost` will use multiprocess pool (`num_workers` works)

    `boost_preds_dir` `model_fraction` `checkpoints_root_dir` `parrot_config_path` `lightgbm_config_path` don't work

- Noise Type
  + `logdis`: Log Normal Noise on Next Request Time (Reuse Distance)
  + `dis`: Normal Noise on Next Request Time (Reuse Distance)
  + `bin`: Binary Flipping Noise on Belady's label (FBP label)
  
- Algorithm

  We implemented most of the existing algorithms in our benchmark, you can also implement and test your own algorithm in the benchmark.

  #### Algorithms and Predictors compatibility

  | Algorithm | PLECO | POPU | Parrot | Pleco-Bin | GBM | Oracle-Dis (Belady) | Oracle-Bin (FBP) |
  |:----------|:------:|:-----:|:----:|:---------:|:---:|:----------:|:----------:|
  | Rand                               | &#10060; | &#10060; | &#10060; | &#10060; | &#10060; | &#10060; | &#10060; |
  | LRU                                | &#10060; | &#10060; | &#10060; | &#10060; | &#10060; | &#10060; | &#10060; |
  | Marker                             | &#10060; | &#10060; | &#10060; | &#10060; | &#10060; | &#10060; | &#10060; |
  | Predict                            | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; |
  | PredictiveMarker<sup>[2]</sup>     | &#10004; | &#10004; | &#10004; | &#10060; | &#10060; | &#10004; | &#10060; |
  | LMarker<sup>[3]</sup>              | &#10004; | &#10004; | &#10004; | &#10060; | &#10060; | &#10004; | &#10060; |
  | LNonMarker<sup>[3]</sup>           | &#10004; | &#10004; | &#10004; | &#10060; | &#10060; | &#10004; | &#10060; |
  | Follower&Robust<sup>[4]</sup>      | &#10004; | &#10004; | &#10004; | &#10060; | &#10060; | &#10004; | &#10060; |
  | Mark0<sup>[5]</sup>                | &#10060; | &#10060; | &#10060; | &#10004; | &#10004; | &#10060; | &#10004; |
  | Mark&Predict<sup>[5]</sup>         | &#10060; | &#10060; | &#10060; | &#10060; | &#10060; | &#10060; | &#10004; |
  | CombineDeterministic<sup>[6]</sup> | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; |
  | CombineRandom<sup>[6]</sup>        | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; |
  | Guard                              | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; |

- Predictor

  + `pleco`: A PLECO Model that gives a page's next request time (reuse distance) when predicting.
  + `popu`: A Popularity Model that gives a page's next request time (reuse distance) when predicting.
  + `parrot`: A Parrot Imitation Model that predicts eviction priority for each page.
  + `pleco-bin`: A PLECO Binary Model based on PLECO that gives a page's **belady's label** when predicting
  + `gbm`: A Gradient Boosting Machine based on Delta and EDC features that gives a page's **belady's label** when predicting
  + `oracle_bin`: An offline predictor that gives the predicted next request time (reuse distance) of a page during prediction, potentially affected by noise (logdis or dis).
  + `oracle_dis`: An offline predictor that gives a page's **belady's label**, potentially affected by noise (logdis, dis or bin).

- Dump and Verbose

  `dump_file`: Enables result dumping, saving test results in the format `$output_root_dir/$dataset/$fraction/$predictor.csv`.

  `verbose`: Enables detailed output. When enabled, all statistical data (hits, misses, hit rates, and competitive ratios) will be displayed; otherwise, only the hit rate and competitive ratio will be shown.
  
- Boost
  
  In `real` mode, `boost` uses a pickle file from `boost_preds_dir` (creating it if it doesn't exist) to accelerate testing.

  In `oracle` mode, boost utilizes a multiprocessing pool to handle all test tasks concurrently.

  We also provide the `boost_fr` parameter, which enables the `Follow&Robust` algorithm to use the Belady trace for calculating the Belady cost within this algorithm. Without this optimization, the entire test process would be significantly slower.
  
- Model Config

  To use the Parrot, GBM, or LRB predictors, you must specify the following parameters (or use the defaults):

  + `model_fraction` (default to 1): Specifies the fraction of the training set to be used for training. (In benchmarking, you must select a fraction corresponding to an existing pre-trained model.)
  + `checkpoints_root_dir`: Checkpoints dir.
  + `parrot_config_path`: The `parrot` predictor's config file.
  + `lightgbm_config_path`: The `gbm` and `lrb` predictors' config file.

  #### LRB-specific parameters

  When using the LRB predictor, you can also specify these additional parameters:

  + `memory_window`: Memory window size controlling feature "forgetting" (default 1000000).
  + `relaxation_factor`: Relaxation factor determining object value influence (default 10.0).
  + `admission_size`: LRB admission queue size, default is associativity/4.
  + `disable_admission`: Set to true to disable LRB admission policy.
  + `disable_edc`: Set to true to disable LRB EDC features.
  + `debug`: Enable debug mode for more detailed output.

  #### Guard+LRB-specific parameters

  Parameters specific to GuardLRB or SimpleGuardLRB:

  + `relax_threshold`: Guard+LRB relaxation threshold for random selection probability (default 0.2).
  + `enable_random_relax`: Enable Guard+LRB random relaxation selection.
  + `phase_reset_percentage`: Guard+LRB phase reset percentage (default 0.7).
  + `guard_weight`: Guard+LRB Guard mechanism weight (0-1), higher values increase protection strength (default 0.7).
  + `simple_relax_times`: SimpleGuardLRB relax_times parameter that controls protection mechanism triggering (default 0).
  + `simple_relax_prob`: SimpleGuardLRB relax_prob parameter that controls protection mechanism probability (default 0.0).

### Model Training

#### GBM/LRB Model

```python
python -m model.lightgbm [--dataset DATASET] [--device DEVICE]
                    [--model_fraction MODEL_FRACTION] [--model_config_path MODEL_CONFIG_PATH]
                    [--checkpoints_root_dir CHECKPOINTS_ROOT_DIR] [--traces_root_dir TRACES_ROOT_DIR]
                    [--iter_threshold] [--real_test] [--lrb_mode]
```

You can use our GBM/LRB training tool to generate and test the cache model.

- Params:
  + `dataset`: Source dataset
  + `device`: Model load device (in GBM/LRB, only `cpu` can work)
  + `model_fraction`: The proportion of the training set used for training. The default value is 1, indicating that the entire training set will be used.
  + `model_config_path`: Model's config
  + `checkpoints_root_dir`: The directory where the checkpoint files are saved after training is completed.
  + `traces_root_dir`: The directory where the datasets are stored.
  + `iter_threshold`: When enabled, this iteratively adjusts the binary splitting threshold from 0 to 1 to determine the optimal threshold for binary classification. If disabled, the threshold defaults to 0.5.
  + `real_test`: After training is completed, the test set will be loaded to evaluate the model's performance.
  + `lrb_mode`: Enable LRB mode, which uses specific LRB features and parameters during training.
  + `memory_window`: Memory window size for LRB mode (default 1000000).
  + `relaxation_factor`: Relaxation factor for LRB mode, affecting Belady value calculation (default 10.0).

- Config Example:
  ```json
  {
      "delta_nums": 10,
      "edc_nums": 10,
      "training": {
          "boosting_type": "gbdt", 
          "objective": "binary", 
          "metric": "l2",
          "learning_rate": 0.01,
          "num_boost_round": 8000,
          "num_leaves": 31, 
          "max_depth": 6,
          "subsample": 0.8, 
          "colsample_bytree": 0.8
      },
      "lrb_config": {
          "memory_window": 1000000,
          "enable_admission": true,
          "enable_edc": true
      }
  }
  ```

  `delta_nums` and `edc_nums`: The dimensions of the **Delta** and **EDC** features, respectively. For more details, refer to [webcachesim2](https://github.com/sunnyszy/lrb) and [Learning relaxed Belady for content distribution network caching](https://dl.acm.org/doi/10.5555/3388242.3388281)<sup>[7]</sup>.

  `training`: The training parameters that will be used by LightGBM.

  `lrb_config`: LRB-specific configuration parameters, including memory window size, whether to enable admission policy, and whether to enable EDC features.

#### Parrot Model

```python
python -m model.parrot [--dataset DATASET] [--device DEVICE]
                    [--model_fraction MODEL_FRACTION] [--model_config_path MODEL_CONFIG_PATH]
                    [--checkpoints_root_dir CHECKPOINTS_ROOT_DIR] [--traces_root_dir TRACES_ROOT_DIR]
```

- Params:
  + `dataset`: Source dataset
  + `device`: Model load device (Parrot use `CUDA_VISIBLE_DEVICES` to set target device)
  + `model_fraction`: The proportion of the training set used for training. The default value is 1, indicating that the entire training set will be used.
  + `model_config_path`: Model's config
  + `checkpoints_root_dir`: The directory where the checkpoint files are saved after training is completed.
  + `traces_root_dir`: The directory where the datasets are stored.
 
- Config Example:
  ```json
  {
    "address_embedder": {
        "embed_dim": 64,
        "max_vocab_size": 5000,
        "type": "dynamic-vocab"
    },
    "cache_line_embedder": "address_embedder",
    "cache_pc_embedder": "none",
    "loss": [
        "ndcg",
        "reuse_dist"
    ],
    "lstm_hidden_size": 128,
    "max_attention_history": 30,
    "pc_embedder": {
        "embed_dim": 64,
        "max_vocab_size": 5000,
        "type": "dynamic-vocab"
    },
    "positional_embedder": {
        "embed_dim": 128,
        "type": "positional"
    },
    "sequence_length": 80,
    "lr": 0.001,
    "total_steps": 25000,
    "eval_freq": 5000,
    "save_freq": 5000,
    "batch_size": 32,
    "collection_multiplier": 5,
    "dagger_init": 1,
    "dagger_final": 1,
    "dagger_steps": 200000,
    "dagger_update_freq": 50000
  }
  ```
  
  You can find more details about Parrot and its training in [cache_replacement](https://github.com/google-research/google-research/tree/master/cache_replacement)<sup>[8]</sup>.

### Script

We also provide some scripts for training and testing, for reference purposes only. You can find them in the `scripts` folder.

## References

[1] Chłędowski, Jakub Polak, Adam Szabucki, Bartosz Zolna, Konrad. 2021. Robust Learning-Augmented Caching: An Experimental Study. 10.48550/arXiv.2106.14693. 

[2] Thodoris Lykouris and Sergei Vassilvitskii. 2021. Competitive Caching with Machine Learned Advice. J. ACM 68, 4, Article 24 (August 2021), 25 pages. https://doi.org/10.1145/3447579

[3] Dhruv Rohatgi. 2020. Near-optimal bounds for online caching with machine-learned advice. In Proceedings of the Thirty-First Annual ACM-SIAM Symposium on Discrete Algorithms (SODA '20). Society for Industrial and Applied Mathematics, USA, 1834–1845.

[4] Sadek, K. A. and Elias, M. Algorithms for caching and mts with reduced number of predictions. arXiv preprint arXiv:2404.06280, 2024.

[5] Antonios Antoniadis, Joan Boyar, Marek Eliáš, Lene M. Favrholdt, Ruben Hoeksma, Kim S. Larsen, Adam Polak, and Bertrand Simon. 2023. Paging with succinct predictions. In Proceedings of the 40th International Conference on Machine Learning (ICML'23), Vol. 202. JMLR.org, Article 39, 952–968.

[6] Wei, A. Better and simpler learning-augmented online caching. arXiv preprint arXiv:2005.13716, 2020.

[7] Zhenyu Song, Daniel S. Berger, Kai Li, and Wyatt Lloyd. 2020. Learning relaxed Belady for content distribution network caching. In Proceedings of the 17th Usenix Conference on Networked Systems Design and Implementation (NSDI'20). USENIX Association, USA, 529–544.

[8] Evan Zheran Liu, Milad Hashemi, Kevin Swersky, Parthasarathy Ranganathan, and Junwhan Ahn. 2020. An imitation learning approach for cache replacement. In Proceedings of the 37th International Conference on Machine Learning (ICML'20), Vol. 119. JMLR.org, Article 579, 6237–6247.
