set -x

aime2024_train_path=$HOME/data/math/train.parquet
aime2024_test_path=$HOME/data/math/test.parquet

train_files="['$aime2024_train_path']"
test_files="['$aime2024_test_path']"

# For async rollout mode, dataset should return raw chat.
rollout_mode="async"
return_raw_chat="True"
chat_scheduler=examples.ppo_trainer.langgraph_chat_scheduler.LangGraphChatCompletionScheduler

python3 -m verl.trainer.main_ppo \
    -cd ../../examples/ppo_trainer \
    algorithm.adv_estimator=gae \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.return_raw_chat=$return_raw_chat \
    data.train_batch_size=4096 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen3-8B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=512 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=6000 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.rollout.chat_scheduler=$chat_scheduler \
    +config@actor_rollout_ref.rollout.langgraph=retool_reflection \
    +actor_rollout_ref.rollout.openai_serving_chat.enable_auto_tools=true \
    +actor_rollout_ref.rollout.openai_serving_chat.tool_parser=hermes \
    +actor_rollout_ref.rollout.enable_thinking=false \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=24000 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=Qwen/Qwen3-8B \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_max_token_len_per_gpu=6000 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_retool' \
    trainer.experiment_name='qwen2.5-7b_retool' \
    trainer.n_gpus_per_node=4 \
    trainer.val_before_train=False \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@
