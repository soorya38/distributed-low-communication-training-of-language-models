#!/bin/sh
# Wall-clock benchmark over an emulated slow link between two containers.
#
# Each container runs 2 worker processes (4 in total). Traffic between the two
# containers goes through tc netem; traffic inside a container does not, like
# fast intra-node and slow inter-node links in a real cluster.
#
#   sh docker/wan_benchmark.sh                      # 100 Mbit/s, 20 ms delay, 300 steps
#   NET_RATE=1gbit NET_DELAY=5ms STEPS=500 sh docker/wan_benchmark.sh
set -e
cd "$(dirname "$0")/.."

export NET_DELAY="${NET_DELAY:-20ms}" NET_RATE="${NET_RATE:-100mbit}"
STEPS="${STEPS:-300}"
mkdir -p results/wan

run() {
    name="$1"; shift
    echo "== $name (delay=$NET_DELAY rate=$NET_RATE, $STEPS steps)"
    TRAIN_ARGS="$* --steps $STEPS --eval-interval 50 --out results/wan/$name.json" \
        docker compose up --abort-on-container-exit --exit-code-from node0 >/dev/null 2>&1
    docker compose down >/dev/null 2>&1
}

docker compose build >/dev/null
run ddp --mode ddp
run diloco_h50 --mode diloco --inner-steps 50
run diloco_h50_q4 --mode diloco --inner-steps 50 --quantize-bits 4

# Give 4-bit DiLoCo the wall-clock time DDP needed (rounded down to whole rounds of 50 steps).
EQUAL_TIME_STEPS=$(python3 -c "import json; d = json.load(open('results/wan/ddp.json'))['seconds']; \
q = json.load(open('results/wan/diloco_h50_q4.json'))['seconds']; print(max(50, int($STEPS * d / q) // 50 * 50))")
BASE_STEPS=$STEPS
STEPS=$EQUAL_TIME_STEPS run diloco_h50_q4_equal_time --mode diloco --inner-steps 50 --quantize-bits 4
STEPS=$BASE_STEPS

# Reference: DDP with no throttling, to show the link is the bottleneck.
NET_DELAY= NET_RATE= run ddp_unthrottled --mode ddp

python3 - <<'EOF'
import json
names = ("ddp_unthrottled", "ddp", "diloco_h50", "diloco_h50_q4", "diloco_h50_q4_equal_time")
rows = {n: json.load(open(f"results/wan/{n}.json")) for n in names}
base = rows["ddp"]["seconds"]
print(f"{'run':<28}{'link':>10}{'steps':>7}{'MB sent':>10}{'wall s':>9}{'vs DDP':>8}{'val loss':>10}")
for n, r in rows.items():
    link = "none" if n == "ddp_unthrottled" else "throttled"
    print(f"{n:<28}{link:>10}{r['steps']:>7}{r['bytes_sent_per_worker'] / 1e6:>10.1f}{r['seconds']:>9.1f}"
          f"{base / r['seconds']:>7.1f}x{r['final_val_loss']:>10.4f}")
EOF
