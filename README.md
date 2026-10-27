# DiLoCo: Distributed Low-Communication Training of Language Models

Build DiLoCo from scratch: workers train locally with AdamW for many steps, then a Nesterov-momentum outer optimizer aggregates pseudo-gradients across rare comm rounds. Implement the model, both optimizers, IID/non-IID sharding, the full DiLoCo round, and compare vs. a synchronous baseline to quantify communication savings.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** init_model_params
- [ ] **2.** relu
- [ ] **3.** model_forward
- [ ] **4.** softmax
- [ ] **5.** cross_entropy_loss
- [ ] **6.** model_backward
- [ ] **7.** init_adamw_state
- [ ] **8.** update_adam_moments
- [ ] **9.** bias_correct_moments
- [ ] **10.** adam_param_step
- [ ] **11.** decoupled_weight_decay
- [ ] **12.** clone_params
- [ ] **13.** scale_params
- [ ] **14.** subtract_params
- [ ] **15.** average_params
- [ ] **16.** iid_shard_dataset
- [ ] **17.** noniid_shard_dataset
- [ ] **18.** sample_worker_batch
- [ ] **19.** local_train_step
- [ ] **20.** inner_train_worker
- [ ] **21.** init_outer_optimizer
- [ ] **22.** update_outer_momentum
- [ ] **23.** nesterov_param_update
- [ ] **24.** compute_outer_gradient
- [ ] **25.** run_diloco_round
- [ ] **26.** train_diloco
- [ ] **27.** train_synchronous_baseline
- [ ] **28.** evaluate_loss
- [ ] **29.** classification_accuracy
- [ ] **30.** communication_savings
