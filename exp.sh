###### Craft world ######

# Train CVAE
python train_cvae.py --net_path=craft_cvae_abstract.pth --mode=train --config craft_cvae.yaml --num_rolls 200

# Our approach
python main.py --mode train --config craft_ours.yaml
python main.py --mode test_stats --config craft_ours.yaml --cvae_path craft_cvae_abstract.pth

# Oracle
python main.py --mode test_stats --config craft_ours.yaml --cvae_path craft_cvae_abstract.pth --oracle

# End to end
python main.py --mode train --config craft_nn.yaml
python main.py --mode test_stats --config craft_nn.yaml --cvae_path craft_cvae_abstract.pth

# World models
python train_world_models.py --mode train_v --config craft_wm.yaml
python train_world_models.py --mode train_m --config craft_wm.yaml
python train_world_models.py --mode train_c --config craft_wm.yaml
python train_world_models.py --mode test_stats --config craft_wm.yaml

# Plot training curves:
python analysis/plots.py --mode training_curve --spec spec_craft.json --save_dir craft --max_iter 400000 --offset 200

# Optimistic
python main.py --mode test_stats --config craft_ours.yaml --optim --test_iter 400000 --cvae_path craft_cvae_abstract.pth

# Visualize examples
python main.py --mode test --cvae_path craft_cvae_abstract.pth --config craft_ours.yaml --train_seed 1 --test_iter 400000 --viz_name viz_test --index_from_sampled 1

# Ablation results
python analysis/plots.py --mode abla --spec spec_abla_craft.json

###### Box world ######

# Train CVAE
python train_cvae.py --net_path=box_cvae_abstract.pth --mode=train --config box_cvae.yaml --num_rolls 20000 --dhid 300 --drep 200

# Our approach
python main.py --mode train --config box_ours.yaml
python main.py --mode test_stats --config box_ours.yaml --cvae_path box_cvae_abstract.pth --dhid 300 --drep 200

# Oracle
python main.py --mode test_stats --config box_ours.yaml --cvae_path box_cvae_abstract.pth --dhid 300 --drep 200 --oracle

# End to end
python main.py --mode train --config box_nn.yaml
python main.py --mode test_stats --config box_nn.yaml --cvae_path box_cvae_abstract.pth --dhid 300 --drep 200

# World models
python train_world_models.py --mode train_v --config box_wm.yaml
python train_world_models.py --mode train_m --config box_wm.yaml
python train_world_models.py --mode train_c --config box_wm.yaml
python train_world_models.py --mode test_stats --config box_wm.yaml

# Relational deep RL
python main.py --mode train --config box_relational.yaml
python main.py --mode test_stats --config box_relational.yaml --cvae_path box_cvae_abstract.pth --dhid 300 --drep 200

# Plot training curves:
python analysis/plots.py --mode training_curve --spec spec_box.json --save_dir box --max_iter 200000 --offset 1000

# Effect of varying number of samples
python analysis/plots.py --mode n_comp --exp box_ours_seed_1 --box

###### Ant-craft ######

# Train goal-following policy
python main_ant.py --mode train --algo sac --env Antg-v1 --eval-episodes 10 --eval-freq 10000
python main_ant.py --mode end --exp Antg-v1_1
python main_ant.py --mode filter --exp Antg-v1_1 --continue_on Antg-v1_1
python main_ant.py --mode train --algo sac --env Antg-v1 --eval-episodes 10 --eval-freq 10000 --continue_on Antg-v1_1

# Our approach
python main.py --mode train --config ant_ours.yaml
python main.py --mode test_stats --config ant_ours.yaml --cvae_path craft_cvae_abstract.pth

# Oracle
python main.py --mode test_stats --config ant_ours.yaml --cvae_path craft_cvae_abstract.pth --oracle

# End to end
python main.py --mode train --config ant_nn.yaml
python main.py --mode test_stats --config ant_nn.yaml --cvae_path craft_cvae_abstract.pth

# World models
python train_world_models.py --mode train_v --config ant_wm.yaml
python train_world_models.py --mode train_m --config ant_wm.yaml
python train_world_models.py --mode train_c --config ant_wm.yaml
python train_world_models.py --mode test_stats --config ant_wm.yaml

# Get stats
python analysis/plots.py --mode seeds --spec spec_ant_seeds.json
