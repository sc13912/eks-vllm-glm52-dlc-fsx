# The stock 0.23.0 DLC ALREADY supports GLM-5.2 (arch GlmMoeDsaForCausalLM, registered in vLLM since
# v0.21.0 via the deepseek_v2 module) — verified against the vLLM supported-models list
# This image adds two things on top:
#   1. Run:AI Model Streamer  -> for --load-format runai_streamer (fast weight load from the FSx PVC)
#   2. Ray                    -> REQUIRED for multi-node vLLM (TP=8 x PP=2 across 2 p5en via LWS).
ARG DLC_BASE
FROM ${DLC_BASE}

# Add the Run:AI Model Streamer (loader) and Ray (multi-node).
# - runai streamer: --no-deps so it can't drag in a different torch/vllm.
# - ray: install directly. NOTE `pip install vllm[ray]==0.23.0` is a no-op here — pip sees vllm already
RUN set -eux; \
    pip install --no-cache-dir --no-deps runai-model-streamer runai-model-streamer-s3; \
    pip install --no-cache-dir "ray[default]>=2.40,<2.50"; \
    python - <<'PY'
import vllm, torch, ray
# 1. Confirm we did NOT disturb the DLC's vLLM / torch.
print("vllm", vllm.__version__, "| torch", torch.__version__, "| ray", ray.__version__)
# 2. Confirm the GLM-5.2 architecture is present in THIS stock image (no upgrade needed).
from vllm.model_executor.models.registry import ModelRegistry
archs = ModelRegistry.get_supported_archs()
assert any("GlmMoeDsa" in a for a in archs), \
    f"GlmMoeDsaForCausalLM missing from stock DLC vLLM {vllm.__version__}; have: {[a for a in archs if 'Glm' in a]}"
# 3. Confirm the runai streamer loader is importable.
import runai_model_streamer  # noqa: F401
print("OK: GLM-5.2 arch present; runai_model_streamer + ray importable")
PY
# 4. Verify the `ray` CLI is on PATH 
RUN which ray && ray --version
