# Program Synthesis Guided Reinforcement Learning for Partially Observed Environments

This repository is the official implementation of *Program Synthesis Guided Reinforcement Learning for Partially Observed Environments*, NeurIPS 2021 spotlight.

## Dependencies

- Python 3.7
- PyTorch 1.7
- [z3-solver 4.8.9](https://pypi.org/project/z3-solver/)
- [MuJoCo simulator](http://www.mujoco.org)
- Open AI gym, [stable baselines3](https://github.com/DLR-RM/stable-baselines3) (included in this repository)
- pygame 2.0.1
- pyyaml 5.3
- numpy 1.18
- matplotlib 3.1.3

## Major experiments

The set of commands for producing the main results of the paper are in `exp.sh`.


### Training CVAE
Command for training the CVAE:
```
python train_cvae.py --net_path=craft_cvae_abstract.pth --mode=train --config craft_cvae.yaml
```

This will save the CVAE to `craft_cvae_abstract.pth`. Change the config flag to `box_cvae.yaml` to train for the box world.

Trained models are at `craft_cvae_abstract.pth` and `box_cvae_abstract.pth`.


### Training the policies
Command for training the policies:
```
python main.py --mode train --config craft_ours.yaml
```

The configurations files for our approach and the baselines for all the benchmarks are saved in [experiments](https://github.com/yycdavid/program-synthesis-guided-RL/tree/main/experiments). Change the `--config` flag to train different models for different benchmarks.

As in the [world models paper](https://arxiv.org/abs/1803.10122), world models are trained stage-by-stage:
```
python train_world_models.py --mode train_v --config box_wm.yaml
python train_world_models.py --mode train_m --config box_wm.yaml
python train_world_models.py --mode train_c --config box_wm.yaml
```


### Testing
Get test results for the save models over the course of training:
```
python main.py --mode test_stats --config craft_ours.yaml --cvae_path craft_cvae_abstract.pth
```

The test set we use is saved in: [craft](https://github.com/yycdavid/program-synthesis-guided-RL/blob/main/resources/craft/test_set.pickle), [box](https://github.com/yycdavid/program-synthesis-guided-RL/blob/main/resources/box/test_set.pickle).

Add `--oracle` flag to evaluate the oracle baseline. Add `--optim` to evaluate with optimistic synthesis; add `--use_random_hallucinator` to evaluate with random hallucinator (ablations).


Plot training curves:
```
python analysis/plots.py --mode training_curve --spec spec_craft.json --save_dir craft --max_iter 400000
```
This will save the training curves to `analysis/craft/`. Change the `--spec` flag to plot for different benchmarks. The spec files for plotting is in `analysis/`.

Save video for a single test run:
```
python main.py --mode test --cvae_path craft_cvae_abstract.pth --config craft_ours.yaml --test_iter 400000 --viz_name viz
```

### Pre-train goal-following policy for Ant
First train a goal-following policy for Ant given random initial poses:
```
python main_ant.py --mode train --algo sac --env Antg-v1 --eval-episodes 10 --eval-freq 10000
```
Then collect the ending states (change `Antg-v1_1` to the folder that the above command save to, if needed):
```
python main_ant.py --mode end --exp Antg-v1_1
```
Then filter:
```
python main_ant.py --mode filter --exp Antg-v1_1 --continue_on Antg-v1_1
```
Then train a policy with initial state randomly sampled from the saved ending states:
```
python main_ant.py --mode train --algo sac --env Antg-v1 --eval-episodes 10 --eval-freq 10000 --continue_on Antg-v1_1
```
Now, the goal-following policy is in `log/sac/EXP_NAME/` (substitute EXP_NAME with the saved folder from the above command). Update `ant_model` and `start_states` paths accordingly in the config files. Then train the policy for Ant-craft:
```
python main.py --mode train --config ant_ours.yaml
```
