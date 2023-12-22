## プロジェクトの構成

```
    ├── .github/           <- 
    │
    ├── data/              <- 
    │
    ├── environments/       <- 
    │
    ├── models/            <-
    │
    ├── notebooks/         <- No use
    │
    ├── outputs/           <- 
    │
    ├── src/               <- 
    │
    ├── tests/             <- No use
    │
    ├── .flake8            <-
    ├── .dockerignore
    ├── .gitignore
    ├── LICENSE
    ├── Makefile           <- 使っていない
    ├── poetry.lock        <- Lock file. DON'T edit this file manually.
    ├── poetry.toml        <- Setting file for Poetry.
    ├── pyproject.toml     <- Setting file for Project. (Poetry, Black, isort, Mypy)
    └── README.md          <- The top-level README for developers.

```
# PC setting
 - OS:Ubuntu20.04LTS
 - CPU:Intel® Core™ i9-10900 CPU @ 2.80GHz × 20 
 - GPU: NVIDIA GeForce RTX 3090
 - Memory: 64GB

# SetUp
## Create accounts
 + install docker
 + create Wandb account from [here](https://www.wandb.jp/)
   + copy user name and API key
 + create GitHub account from [here](https://github.co.jp/)
 + create figshare account from [here](https://figshare.com/)

## Download code from github
```shell
UserNmae@yourPC: $ git clone git@github.com:tohsato-lab/S-loss.git
```

## Create docker container and install python3 virtual environment 
```shell
UserNmae@yourPC:S-loss $ cd environment/gpu
UserNmae@yourPC:S-loss $ docker compose up -d
UserNmae@yourPC:S-loss $ docker compose exec core bash
challenger@hoge:~/ascender $ poetry install
```
## Wandb setting
 + wandb setting in docker container
``` shell
challenger@hoge:~/ascender $ poetry run wandb init
Let's setup this directory for W&B!
wandb: Logging into wandb.ai. (Learn how to deploy a W&B server locally: https://wandb.me/wandb-server)
wandb: You can find your API key in your browser here: https://wandb.ai/authorize
wandb: Paste an API key from your profile and hit enter, or press ctrl+c to quit: {API key} 
wandb: Which team should we use?
wandb: (1) {Youre User Name}
wandb: (2) Manual entry
wandb: Enter your choice: 1
wandb: Which project should we use?
wandb: (1) Create New
wandb: Enter your choice: 1
wandb: You chose 'Create New'
Enter a name for your new project: {Project Name}
This directory is configured!  Next, track a run:
challenger@hoge:~/ascender $ poetry run wandb init
```
# Downloading datasets
+ downloading from [here](URL)

# Running
## Running Train and Test
 1. Edit main.py
``` python {main.py}
if __name__ == '__main__':
    CFG = get_args()
    if CFG.wandb_logging:
        run = wandb.init(
            project="{Project Name}",   # ← here
            config={k: v for k, v in dict(vars(CFG)).items() if '__' not in k},
            name=f"{CFG.Name}",
            entity="{User Name}"        # ← here
        )
        wandb.save(os.path.join(CFG.Homepath, 'src', 'train_core.py'))
    main_loop(CFG)
```
 2. Running main.py (Train and Test)
```shell
challenger@hoge:~/ascender $ poetry run python3 main.py
```

## Running Prediction and Create BD5 dataset
 1. download model weight (hogehoge.pth) from wandb and save in " **models/** " folder
 2. Edit prediction.py
```python
model = load_model.load_deplaboV3plus(configs=configs)
model_name = f'model_weight_name'                       # ←here
model_path = os.path.join(configs.Homepath, "models", model_name + '.pth')
model.load_state_dict(torch.load(model_path))
```
 3. Running prediction.py
```shell
challenger@hoge:~/ascender $ poetry run python3 predinction.py
```
 4. Resizing and Create BD5
```shell
challenger@hoge:~/ascender $ cd data
challenger@hoge:~/ascender/data $ poetry run python3 resize.py
challenger@hoge:~/ascender/data $ poetry run python3 create_h5_split_class_id.py
```