# GLM-5.2 on Amazon EKS — 2× p5en.48xlarge (vLLM + LWS + EFA + FSx for Lustre)

Runbook + manifests to serve **GLM-5.2-FP8** (~705 GB, 141 shards) across **2× p5en.48xlarge**
(16× H200) on Amazon EKS, using GPUs from a temporary **EC2 Capacity Block for ML**.

Topology: **TP=8** intra-node (NVLink) × **PP=2** inter-node (EFA / NCCL / GPUDirect RDMA), served by
**vLLM + LeaderWorkerSet + Ray**, weights on **FSx for Lustre**, loaded fast with
`--load-format runai_streamer` (no GDS, no custom AMI). Served at the full **1M-token context window**,
internal-only (ClusterIP + `kubectl port-forward` — no internet-facing load balancer).


---

## Per-step guides

| Step | Dir | What |
|---|---|---|
| 1 | [01-cluster/](01-cluster/) | Provision EKS cluster (control plane + CPU system NG), 2-AZ VPC |
| 2 | [02-fsx/](02-fsx/) | FSx for Lustre (EFA, PERSISTENT_2, **4 OSTs**) + CSI + PV/PVC |
| 3 | [03-nodegroup/](03-nodegroup/) | Capacity-Block p5en node group + NVIDIA device plugin |
| 4 | [04-efa-validation/](04-efa-validation/) | NCCL `all_reduce_perf` EFA/GPUDirect test |
| 5 | [05-image/](05-image/) | vLLM image (stock DLC + Run:AI streamer + Ray) |
| 6 | [06-weights/](06-weights/) | Stage GLM-5.2-FP8 weights → FSx (auto-striped) |
| 7 | [07-serving/](07-serving/) | Serve with vLLM + LeaderWorkerSet (TP=8 × PP=2, 1M context) |

---

## ⚠️ Before you run anything

- These manifests create **real, expensive** AWS resources. Nothing runs automatically — each step is a
  deliberate command.
- Fill in every `<PLACEHOLDER>` (account ID, Capacity Block ID, HF token). The `notes-summary.txt` shows
  the exact env exports each step needs.
- Everything pins to the **single AZ of your Capacity Block** (`<CB_AZ>`): the FSx file
  system and the p5en node group both live there. The cluster VPC spans two AZs. 
  control plane requires subnets in ≥2 AZs.

---

## Prerequisites (tools)

- `awscli` v2, `eksctl` **≥ v0.205.0** (native Capacity-Block support), `kubectl`, `helm`,
  `docker` (x86 build host)
- An IAM principal allowed to create EKS / EC2 / FSx / IAM resources (least-privilege; ReadOnly for
  inspection)
- A Hugging Face token with access to `zai-org/GLM-5.2-FP8`

---

## What can run before the Capacity Block is active

You can do everything **except** the GPU-dependent steps ahead of the reservation window:

- **Ahead of the CB:** Step 1 (cluster), Step 2 (FSx), Step 5 (build/push image), Step 6 (download
  weights), and *creating* the Step-3 node group at `desiredCapacity: 0`.
- **Requires live GPUs:** scaling the node group 0→2, Step 4 (NCCL), and Step 7 (serve).

---

## Common environment

```bash
export AWS_REGION="ap-northeast-2"; export REGION="$AWS_REGION"
export ACCOUNT_ID="<ACCOUNT_ID>"
export CLUSTER_NAME="glm52-poc-cluster"
export K8S_VERSION="1.36"
export ECR_REPO="glm52-vllm"; export IMAGE_TAG="glm52-fp8"
export IMAGE_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}"
export DLC_BASE="763104351884.dkr.ecr.${REGION}.amazonaws.com/vllm:0.23.0-gpu-py312-cu130-ubuntu22.04-ec2"
export HF_MODEL="zai-org/GLM-5.2-FP8"
export MODEL_DIR="/mnt/fsx/glm-5.2-fp8"
export HF_TOKEN="hf_xxxx"     # rotate after the run
```

---

## Step 1 — Empty cluster

```bash
envsubst < 01-cluster/cluster.yaml | eksctl create cluster -f -
aws eks update-kubeconfig --name "$CLUSTER_NAME" --region "$REGION"
kubectl get nodes         # only the CPU system nodes so far
```

## Step 2 — FSx for Lustre (4 OSTs)

> **Sizing = OST count.** At the 1000 MB/s/TiB tier, 1 OST = 4.8 TiB, and **OST count is the real lever
> for weight-load speed**. We use **19.2 TiB = 4 OSTs** as a balanced baseline. `FSX_CAPACITY_GIB` sets it.

```bash
# 2a. Create a dedicated SG (self all-traffic INGRESS *and* EGRESS + Lustre ports 988/1018-1023 from the
#     node SG) BEFORE create-file-system — FSx validates network settings at creation.

# Look up the cluster's VPC + node/primary SG (both auto-created by eksctl in Step-1):
export VPC_ID="$(aws eks describe-cluster --name "$CLUSTER_NAME" --region "$REGION" \
  --query 'cluster.resourcesVpcConfig.vpcId' --output text)"

export NODE_SG_ID="$(aws eks describe-cluster --name "$CLUSTER_NAME" --region "$REGION" \
  --query 'cluster.resourcesVpcConfig.clusterSecurityGroupId' --output text)"

echo "VPC_ID=$VPC_ID  NODE_SG_ID=$NODE_SG_ID"

# Create the dedicated FSx SG in the cluster VPC:
export FSX_SG_ID="$(aws ec2 create-security-group --region "$REGION" \
  --group-name glm52-fsx-lustre-sg --description 'GLM-5.2 PoC FSx Lustre' --vpc-id "$VPC_ID" \
  --tag-specifications 'ResourceType=security-group,Tags=[{Key=auto-delete,Value=no},{Key=project,Value=glm52-poc}]' \
  --query 'GroupId' --output text)"
echo "FSX_SG_ID=$FSX_SG_ID"

# self all-traffic — BOTH directions. The EGRESS rule is the one people miss -> InvalidNetworkSettings.
aws ec2 authorize-security-group-ingress --group-id "$FSX_SG_ID" --protocol -1 --source-group "$FSX_SG_ID" --region "$REGION"
aws ec2 authorize-security-group-egress  --group-id "$FSX_SG_ID" --protocol -1 --source-group "$FSX_SG_ID" --region "$REGION"
# Lustre ports from the node SG
aws ec2 authorize-security-group-ingress --group-id "$FSX_SG_ID" --protocol tcp --port 988       --source-group "$NODE_SG_ID" --region "$REGION"
aws ec2 authorize-security-group-ingress --group-id "$FSX_SG_ID" --protocol tcp --port 1018-1023 --source-group "$NODE_SG_ID" --region "$REGION"

# 2b. Create an EFA-enabled PERSISTENT_2 SSD file system in the CB's AZ

export FSX_CAPACITY_GIB="19200"       # 19.2 TiB => 4 OSTs (~19 GB/s aggregate). See SIZING NOTE below.
export FSX_THROUGHPUT_TIER="1000"     # MB/s per TiB (125/250/500/1000); 1000 = most OST-dense
export CB_AZ="<CB_AZ>"        # the CB's AZ — FSx + p5en nodes both live here

export CB_SUBNET_ID="$(aws ec2 describe-subnets --region "$REGION" \
  --filters "Name=vpc-id,Values=$VPC_ID" "Name=availability-zone,Values=$CB_AZ" \
            "Name=tag:kubernetes.io/role/internal-elb,Values=1" \
  --query 'Subnets[0].SubnetId' --output text)"
echo "CB_SUBNET_ID=$CB_SUBNET_ID"   # private subnet in $CB_AZ (eksctl tags private subnets internal-elb=1)

aws fsx create-file-system --region "$REGION" \
  --file-system-type LUSTRE --storage-type SSD \
  --storage-capacity "$FSX_CAPACITY_GIB" \
  --subnet-ids "$CB_SUBNET_ID" --security-group-ids "$FSX_SG_ID" \
  --lustre-configuration '{
      "DeploymentType":"PERSISTENT_2",
      "PerUnitStorageThroughput":'"$FSX_THROUGHPUT_TIER"',
      "EfaEnabled":true,
      "MetadataConfiguration":{"Mode":"AUTOMATIC"},
      "DataCompressionType":"NONE"
    }' \
  --tags Key=auto-delete,Value=no Key=project,Value=glm52-poc


# Wait for AVAILABLE (~10-15 min), then capture the id + mount name:
aws fsx describe-file-systems --region "$REGION" \
  --query 'FileSystems[-1].[FileSystemId,LustreConfiguration.MountName,Lifecycle]' --output table


# 2c. Install the FSx CSI driver, then bind a static PV/PVC:

export FSX_ID="<FSX_ID>"              # from 2b
export FSX_MOUNT_NAME="<MOUNT_NAME>"  # from 2b
export FSX_DNS="${FSX_ID}.fsx.${REGION}.amazonaws.com"


helm repo add aws-fsx-csi-driver https://kubernetes-sigs.github.io/aws-fsx-csi-driver/ && helm repo update
helm upgrade --install aws-fsx-csi-driver aws-fsx-csi-driver/aws-fsx-csi-driver -n kube-system
envsubst < 02-fsx/fsx-pv.yaml  | kubectl apply -f -
envsubst < 02-fsx/fsx-pvc.yaml | kubectl apply -f -


# 2d. Verify: kubectl apply -f 02-fsx/fsx-test-pod.yaml ; kubectl exec -it fsx-test -- lfs df -h /mnt/fsx
```

## Step 3 — CB-backed p5en node group (create at 0)

```bash
export CB_RESERVATION_ID="<CB_RESERVATION_ID>"   # the ACTIVE Capacity Block reservation id (cr-...)
export CB_AZ="<CB_AZ>"                   # the CB's AZ (already exported in Step-2)
eksctl create nodegroup -f <(envsubst < 03-nodegroup/nodegroup.yaml)

# ... when the CB is ACTIVE, scale up:
eksctl scale nodegroup --cluster "$CLUSTER_NAME" --name p5en-cb --nodes 2 --nodes-max 2 --region "$REGION"

# Make GPUs schedulable + verify allocatable (expect gpu=8, efa=16 per node):
kubectl apply -f 03-nodegroup/nvidia-device-plugin.yaml
```

## Step 4 — Validate EFA/NCCL (before serving)

```bash
# Install the Kubeflow MPI Operator once — MUST be --server-side (the MPIJob CRD schema exceeds the
# client-side 256 KB annotation limit):
kubectl apply --server-side -f https://raw.githubusercontent.com/kubeflow/mpi-operator/v0.8.0/deploy/v2beta1/mpi-operator.yaml
kubectl apply -f 04-efa-validation/nccl-test-mpijob.yaml
kubectl logs -f <nccl-tests-launcher-pod>
```
> Expect `NCCL INFO NET/OFI Selected Provider is efa` (EFA, not TCP fallback), `busbw` plateauing at the
> large-message tail, and `0 wrong`. Reference: ~477 GB/s busbw across the 2 nodes.

## Step 5 — Thin vLLM image (stock DLC + Run:AI streamer + Ray)

The stock **vLLM 0.23.0 DLC already supports GLM-5.2** (`GlmMoeDsaForCausalLM`, in vLLM since v0.21.0), so
we don't upgrade vLLM or torch. The image adds only: (1) the **Run:AI Model Streamer** for
`--load-format runai_streamer`, and (2) **Ray**, which multi-node vLLM requires

```bash
# Create the ECR repo (idempotent; tagged for cleanup):
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin 763104351884.dkr.ecr.${REGION}.amazonaws.com
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

aws ecr create-repository --repository-name "$ECR_REPO" --region "$REGION" \
  --image-scanning-configuration scanOnPush=true \
  --tags Key=auto-delete,Value=no Key=project,Value=glm52-poc 2>/dev/null || true

# Build and push:
docker build --platform linux/amd64 --build-arg DLC_BASE="$DLC_BASE" -t "$IMAGE_URI" 05-image/
docker push "$IMAGE_URI"
```

## Step 6 — Stage weights on FSx

```bash
kubectl create secret generic hf-token --from-literal=token="$HF_TOKEN" -n default
envsubst '${MODEL_DIR} ${HF_MODEL}' < 06-weights/download-weights-job.yaml | kubectl apply -f -
kubectl get job glm52-weight-download -w      # approx. ~13 min; the Job verifies all 141 shards
```
> No `lfs setstripe` needed — verify FSx auto-striped across all 4 OSTs:
> `kubectl exec fsx-test -- lfs df -h /mnt/fsx`
> `kubectl exec fsx-test -- lfs getstripe -c /mnt/fsx/glm-5.2-fp8/model-00001-of-00141.safetensors` → `4`.

## Step 7 — Serve

```bash
helm install lws oci://registry.k8s.io/lws/charts/lws --version 0.9.0 -n lws-system --create-namespace --wait
envsubst '$IMAGE_URI $MODEL_DIR' < 07-serving/lws-serving.yaml | kubectl apply -f -
kubectl apply -f 07-serving/service.yaml
kubectl get pods -l app=glm52 -w              # glm52-0 (leader) + glm52-0-1 (worker); wait glm52-0 = 1/1
kubectl logs -f glm52-0
```
> - Serves at `--max-model-len 1048576` (GLM-5.2's full 1M window; the 2-node KV pool of ~2.5M tokens
>   leaves ~2.2× concurrency at 1M — lower `--max-model-len` to trade context for concurrency).
> - `VLLM_CACHE_ROOT` points at FSx so the torch.compile + DeepGEMM caches persist across restarts (cold
>   init ~375 s → warm ~60 s).

### Test
```bash

kubectl exec fsx-test -- curl -sS http://glm52-leader.default.svc.cluster.local:8000/v1/chat/completions -H content-type:application/json \
-d '{"model":"glm-5.2-fp8","messages":[{"role":"user","content":"What is Kubernetes? Answer in 3 sentences."}],"max_tokens":300,"temperature":0.7}' | python3 -m json.tool

kubectl port-forward svc/glm52-leader 8000:8000 &
curl localhost:8000/v1/models
curl localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"glm-5.2-fp8","messages":[{"role":"user","content":"In one sentence, what is tensor parallelism?"}]}'

# ---- Benchmark (run from the LEADER pod)  ----
kubectl exec glm52-0 -- bash -lc 'vllm bench serve --backend openai-chat --model /mnt/fsx/glm-5.2-fp8 \
--served-model-name glm-5.2-fp8 --tokenizer /mnt/fsx/glm-5.2-fp8 \
--base-url http://localhost:8000 --endpoint /v1/chat/completions \
--dataset-name random --num-prompts 200 --max-concurrency 64 \
--random-input-len 1024 --random-output-len 256'


```

---

## Results

**Throughput** (2 nodes, `vllm bench serve`, 200 prompts, concurrency 64, 1024-in/256-out): ~7,900 tok/s
total, ~1,570 tok/s output, median TTFT ~274 ms, median ITL ~35 ms.

**Networking:** NCCL `all_reduce_perf` reached ~477 GB/s busbw across the two nodes, 0 wrong — EFA
GPUDirect RDMA confirmed active.

**Storage / cold start** (weight-load of 705 GB, page cache dropped each run):

| FSx OSTs | Capacity | Avg weight-load | Full cold start (warm caches) |
|---|---|---|---|
| 2 | 9.6 TiB | ~676 s | ~840 s |
| **4** | **19.2 TiB** | **~382 s** | **~540 s** |
| 6 | 28.8 TiB | ~274 s | ~400 s |

- Weight-load scales ~linearly with OST count (~0.5 GB/s per OST); diminishing but real returns past 4.
- Run:AI streamer concurrency/chunk/memory tuning is ~flat at a given OST count
- Persisting the compile/kernel caches on FSx (`VLLM_CACHE_ROOT`) turns a ~375 s cold engine-init into
  ~60 s warm on subsequent starts.

---

## Teardown

```bash
kubectl delete lws glm52 ; kubectl delete svc glm52-leader
kubectl delete job glm52-weight-download ; kubectl delete secret hf-token
kubectl delete pvc fsx-glm52-pvc ; kubectl delete pv fsx-glm52-pv
eksctl scale nodegroup --cluster "$CLUSTER_NAME" --name p5en-cb --nodes 0 --region "$REGION"   # if not already drained
aws fsx delete-file-system --file-system-id "$FSX_ID" --region "$REGION"
eksctl delete cluster --name "$CLUSTER_NAME" --region "$REGION"
```
---
