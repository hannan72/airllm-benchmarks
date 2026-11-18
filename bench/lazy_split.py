"""Layer-by-layer splitter that never loads a whole checkpoint shard.

AirLLM's ``split_and_save_layers`` reads each shard with ``safetensors.torch.load_file``, i.e. the
whole shard at once. ``google/gemma-4-26B-A4B-it`` ships a single 49.9 GB shard, which fails
on a 31 GB-RAM machine ("unable to mmap 49907246508 bytes ... Cannot allocate memory"). And when
AirLLM streams shards from the Hub it has to hold that whole file on disk, which overflowed Kaggle's
scratch quota.

This splitter parses the safetensors header and reads only the byte ranges of one layer at a
time, either from a locally cached shard or directly from the Hub over HTTP range requests
(``HfFileSystem``), so neither the shard nor the checkpoint ever has to fit in RAM or on disk.
It writes exactly the files AirLLM expects (``<layer>.safetensors`` + ``.done`` markers in
``splitted_model[.4bit]``), compressing each layer with AirLLM's own (patched) compressor, so
AirLLM finds the split and skips its own splitter.
"""

from __future__ import annotations

import json
import struct
import time
from collections import defaultdict
from pathlib import Path

import torch

_DTYPES = {
    "BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64,
    "I8": torch.int8, "U8": torch.uint8, "I16": torch.int16, "I32": torch.int32, "I64": torch.int64,
    "BOOL": torch.bool, "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2,
}


def _read_header(f):
    (n,) = struct.unpack("<Q", f.read(8))
    header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header, 8 + n


def _open_shard(repo_id, filename, revision=None):
    """Local cached file if present, else a random-access HTTP file on the Hub."""
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(repo_id, filename, revision=revision, local_files_only=True)
        return open(path, "rb"), "local"
    except Exception:  # noqa: BLE001 - not cached
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        rev = f"@{revision}" if revision else ""
        return fs.open(f"{repo_id}{rev}/{filename}", "rb", block_size=0), "remote"


def _layer_list(layer_names, keys):
    from airllm.utils import cpu_resident_module_names, layer_owner

    prefix = layer_names["layer_prefix"]
    n_layers = len({int(k[len(prefix):].split(".")[1]) for k in keys if prefix in k})
    layers = [layer_names["embed"]] + [f"{prefix}.{i}" for i in range(n_layers)] + \
             [layer_names["norm"], layer_names["lm_head"]]
    if "rotary_pos_emb" in layer_names:
        layers = [layer_names["rotary_pos_emb"]] + layers
    layers += list(layer_names.get("resident", []))
    layers += cpu_resident_module_names(layer_names, keys)
    layers = [l + "." for l in layers]
    layers = [l for l in layers if any(k.startswith(l) for k in keys)]
    owned = {layer_owner(k, layers) for k in keys}
    return [l for l in layers if l in owned], layer_owner


def lazy_split(repo_id, layers_root, layer_names, compression=None, revision=None, log=print):
    """Split ``repo_id`` into AirLLM per-layer files under ``layers_root``. Returns stats."""
    import airllm.utils as airllm_utils
    from airllm.persist import ModelPersister
    from huggingface_hub import hf_hub_download

    saving_path = Path(layers_root) / ("splitted_model" + (f".{compression}" if compression else ""))
    saving_path.mkdir(parents=True, exist_ok=True)
    persister = ModelPersister.get_model_persister()

    index_file = hf_hub_download(repo_id, "model.safetensors.index.json", revision=revision)
    weight_map = json.load(open(index_file))["weight_map"]
    layers, layer_owner = _layer_list(layer_names, list(weight_map))
    todo = [l for l in layers if not persister.model_persist_exist(l, saving_path)]
    if not todo:
        log(f"[lazy_split] all {len(layers)} layer files already present in {saving_path}")
        return {"layers": len(layers), "bytes_read": 0, "sources": []}

    expected = defaultdict(int)  # layer -> number of tensors it owns
    owner_of = {}
    for k in weight_map:
        o = layer_owner(k, layers)
        if o is not None:
            owner_of[k] = o
            expected[o] += 1

    pending = defaultdict(dict)
    bytes_read, sources = 0, []
    t0 = time.time()
    for shard in sorted(set(weight_map.values())):
        f, where = _open_shard(repo_id, shard, revision)
        sources.append(where)
        header, data_start = _read_header(f)
        items = sorted(((k, v) for k, v in header.items() if owner_of.get(k) in todo),
                       key=lambda kv: kv[1]["data_offsets"][0])
        log(f"[lazy_split] {shard} ({where}): {len(items)} tensors for {len({owner_of[k] for k, _ in items})} layers")
        for key, meta in items:
            start, end = meta["data_offsets"]
            f.seek(data_start + start)
            buf = bytearray(f.read(end - start)) if end > start else bytearray()
            bytes_read += end - start
            dtype = _DTYPES[meta["dtype"]]
            t = torch.frombuffer(buf, dtype=dtype) if buf else torch.empty(0, dtype=dtype)
            layer = owner_of[key]
            pending[layer][key] = t.reshape(meta["shape"])
            if len(pending[layer]) == expected[layer]:
                state = airllm_utils.compress_layer_state_dict(pending.pop(layer), compression)
                persister.persist_model(state, layer, saving_path)
                del state
        f.close()
    if pending:
        raise RuntimeError(f"incomplete layers after reading all shards: {sorted(pending)[:5]}")
    dt = time.time() - t0
    log(f"[lazy_split] wrote {len(todo)} layer files, read {bytes_read / 1e9:.1f} GB in {dt:.0f} s "
        f"({bytes_read / 1e9 / max(dt, 1e-9):.2f} GB/s)")
    return {"layers": len(layers), "bytes_read": bytes_read, "seconds": dt, "sources": sorted(set(sources))}
