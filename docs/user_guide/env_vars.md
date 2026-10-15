# Environment Variables

Spyre Inference reads the following `SPYRE_*` environment variables to configure
the plugin. Each is evaluated the first time it is read, so set it before
launching vLLM. The list below is generated from `spyre_inference/envs.py`; the
comment above each entry documents its effect and default.

```python
--8<-- "spyre_inference/envs.py:env-vars-definition"
```

Spyre Inference also sets `TORCH_SPYRE_CACHE_SCAN=1` unless it is already set. This torch-spyre variable lets the Inductor FX graph and AOTAutograd caches keep graphs that use the `scan` op, which `for_each_tile` attention compiles to, so a restart reuses the compiled attention graphs instead of compiling them again. Set `TORCH_SPYRE_CACHE_SCAN=0` to turn this off.
