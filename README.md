# diloco-distributed-low-communication-training-of-language-models
Build DiLoCo from scratch: workers train locally with AdamW for many steps, then a Nesterov-momentum outer optimizer aggregates pseudo-gradients across rare comm rounds. Implement the model, both optimizers, IID/non-IID sharding, the full DiLoCo round, and compare vs. a synchronous baseline to quantify communication savings.
