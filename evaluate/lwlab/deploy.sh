python evaluate/deploy_batched.py \
    --model_path experiment/meanflow_4090/checkpoint-35000 \
    --data_stats_path /home/johndoe/Documents/lerobot-hilserl/data/lerobot_lift_visual_processed_280/meta/stats.json \
    --cfg_path /home/johndoe/Documents/AgiBot-World/go1/configs/go1_air_sft_lerobot_meanflow.py \
    --host 127.0.0.1 \
    --port 8000 \
    --batch_size 20

python '/home/johndoe/Documents/AgiBot-World/evaluate/lwlab/eval_lwlab.py' \
    --config_path /home/johndoe/Documents/lerobot-hilserl/src/lerobot/configs/rl/hilserl_sim_lwlab_lerobot/train_lwlab_hil_lerobotPnP_w_data.json \
    --host 127.0.0.1 \
    --port 8000 \
    --n_steps 200

