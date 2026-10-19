"""Auditable native RAW transport patch for the pinned LIBERO author client.

Edits only a generated client copy. Original author code and historical clients
stay unchanged. Anchor drift is an error, never a best-effort partial patch.
"""

from __future__ import annotations

# ruff: noqa: E501


def _once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"native transport anchor changed: {old[:90]!r}")
    return source.replace(old, new, 1)


def patch_native_libero_client(source):
    source = _once(
        source,
        "            raw_frame_history = None\n",
        "            raw_frame_history = None\n            raw_frame_id_history = None\n",
    )
    source = _once(
        source,
        "                        raw_burst_images = []\n",
        "                        raw_burst_images = []\n"
        "                        raw_burst_ids = []\n",
    )
    source = _once(
        source,
        "                            history.append(current_raw)\n",
        """                            history.append(current_raw)
                            if raw_frame_id_history is None:
                                raw_frame_id_history = [
                                    collections.deque(maxlen=burst_frames) for _ in raw_sources
                                ]
                            raw_frame_id_history[view_idx].append(policy_step)
                            ids = list(raw_frame_id_history[view_idx])
                            ids = [ids[0]] * (burst_frames - len(ids)) + ids
                            raw_burst_ids.append(np.asarray(ids, dtype=np.int64))
""",
    )
    source = _once(
        source,
        """                            current_raw = image_tools.convert_to_uint8(
                                image_tools.resize_image(current_raw, args.resize_size, args.resize_size)
                            )""",
        """                            if client.metadata.get("owac_native_transport", False):
                                if args.rawvla_bench_representation != "raw":
                                    raise ValueError("OWAC requires the native RAW representation")
                                current_raw = np.ascontiguousarray(current_raw)
                            else:
                                current_raw = image_tools.convert_to_uint8(
                                    image_tools.resize_image(current_raw, args.resize_size, args.resize_size)
                                )""",
    )
    source = _once(
        source,
        '                            example["raw_burst_float32"] = raw_burst_images\n',
        """                            example["raw_burst_float32"] = raw_burst_images
                            if client.metadata.get("owac_native_transport", False):
                                example["raw_frame_ids"] = raw_burst_ids
                                example["episode_id"] = f"{args.task_suite}/{task_id}/{ep_idx}"
""",
    )
    source = _once(
        source,
        '        payload = {"examples": [model_input]}\n',
        """        for key in ("raw_frame_ids", "episode_id"):
            if key in example:
                model_input[key] = example[key]
        payload = {"examples": [model_input]}
""",
    )
    compile(source, "owac_native_libero_client.py", "exec")
    return source
