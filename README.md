# DiLoCo: Distributed Low-Communication Training of Language Models

Build DiLoCo from scratch: workers train locally with AdamW for many steps, then a Nesterov-momentum outer optimizer aggregates pseudo-gradients across rare comm rounds. Implement the model, both optimizers, IID/non-IID sharding, the full DiLoCo round, and compare vs. a synchronous baseline to quantify communication savings.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** init_model_params
- [x] **2.** relu
- [x] **3.** model_forward
- [x] **4.** softmax
- [x] **5.** cross_entropy_loss
- [x] **6.** model_backward
- [x] **7.** init_adamw_state
- [x] **8.** update_adam_moments
- [x] **9.** bias_correct_moments
- [x] **10.** adam_param_step
- [x] **11.** decoupled_weight_decay
- [x] **12.** clone_params
- [x] **13.** scale_params
- [x] **14.** subtract_params
- [x] **15.** average_params
- [x] **16.** iid_shard_dataset
- [x] **17.** noniid_shard_dataset
- [x] **18.** sample_worker_batch
- [x] **19.** local_train_step
- [x] **20.** inner_train_worker
- [x] **21.** init_outer_optimizer
- [x] **22.** update_outer_momentum
- [x] **23.** nesterov_param_update
- [x] **24.** compute_outer_gradient
- [x] **25.** run_diloco_round
- [x] **26.** train_diloco
- [ ] **27.** train_synchronous_baseline
- [ ] **28.** evaluate_loss
- [ ] **29.** classification_accuracy
- [ ] **30.** communication_savings
