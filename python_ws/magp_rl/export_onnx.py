#!/usr/bin/env python3
"""
Export trained PPO/SAC actor weights from Flax checkpoints to ONNX.

Examples:
  # Flat input (default): obs[batch, obs_dim]
  python3 export_onnx.py \
    --agent sac \
    --checkpoint-dir ./ckpts/train/2026-03-16/12-34-56 \
    --output ./ckpts/train/2026-03-16/12-34-56/sac_actor.onnx

  # Isaac-style input: scan_input[batch, 1, points] with in-graph normalization
  python3 export_onnx.py \
    --agent sac \
    --checkpoint-dir ./ckpts/train/2026-03-16/12-34-56 \
    --lidar-profile hokuyo \
    --input-layout scan \
    --normalize-input \
    --input-name scan_input \
    --output-name control_output
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _parse_args():
    parser = argparse.ArgumentParser(description="Export Flax actor checkpoint to ONNX")
    parser.add_argument("--agent", type=str, choices=["ppo", "sac", "td3"], required=True)
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step (optional)")
    parser.add_argument("--output", type=str, default=None, help="Output .onnx path")
    parser.add_argument("--obs-dim", type=int, default=None, help="Model input dim used at training time")
    parser.add_argument("--action-dim", type=int, default=2)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--sac-output",
        type=str,
        choices=["deterministic", "mean_logstd", "all"],
        default="deterministic",
        help="SAC ONNX outputs",
    )
    parser.add_argument("--input-layout", type=str, choices=["flat", "scan"], default="flat")
    parser.add_argument(
        "--scan-points",
        type=int,
        default=None,
        help="Input points when --input-layout=scan. Default: obs_dim",
    )
    parser.add_argument("--normalize-input", action="store_true", help="Apply clip/div normalization in ONNX graph")
    parser.add_argument("--max-lidar-range", type=float, default=None, help="Normalization range for lidar clip/div")
    parser.add_argument(
        "--lidar-profile",
        type=str,
        choices=["custom", "hokuyo", "t_mini_plus"],
        default="custom",
        help="Preset for obs_dim/scan_points/max_lidar_range defaults",
    )
    parser.add_argument("--lidar-fov-rad", type=float, default=None, help="Optional metadata only")
    parser.add_argument("--input-name", type=str, default="obs", help="ONNX input tensor name")
    parser.add_argument("--output-name", type=str, default="action", help="Deterministic action output name")
    parser.add_argument("--mean-output-name", type=str, default="mean", help="SAC mean output name")
    parser.add_argument("--logstd-output-name", type=str, default="log_std", help="SAC log_std output name")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run a lightweight ONNXRuntime inference check if available",
    )
    return parser.parse_args()


def _import_runtime_deps():
    try:
        import jax  # noqa: F401
        from flax.training import checkpoints  # noqa: F401
        import onnx  # noqa: F401
        from onnx import helper, numpy_helper, TensorProto  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing dependency: {exc.name}. "
            "Install required packages in this venv (jax/flax/onnx)."
        )


def _resolve_ckpt_dir_and_step(path_str: str, step: int | None):
    ckpt_path = Path(path_str)
    inferred_step = step
    ckpt_dir = ckpt_path
    if ckpt_path.name.startswith("checkpoint_"):
        if inferred_step is None:
            token = ckpt_path.name.split("_")[-1]
            if token.isdigit():
                inferred_step = int(token)
        ckpt_dir = ckpt_path.parent
    return ckpt_dir, inferred_step


def _to_plain_dict(tree: Any):
    try:
        from flax.core import FrozenDict
        from flax.core.frozen_dict import unfreeze

        if isinstance(tree, FrozenDict):
            tree = unfreeze(tree)
    except Exception:
        pass

    if isinstance(tree, Mapping):
        return {k: _to_plain_dict(v) for k, v in tree.items()}
    return tree


def _as_numpy(x):
    import jax

    return np.asarray(jax.device_get(x), dtype=np.float32)


def _sorted_layer_keys(module: Mapping[str, Any], prefix: str):
    keys = [k for k in module.keys() if k.startswith(f"{prefix}_")]

    def _idx(name: str):
        try:
            return int(name.rsplit("_", 1)[-1])
        except ValueError:
            return 10**9

    return sorted(keys, key=_idx)


def _extract_actor_param_tree(restored: Mapping[str, Any]):
    if "actor_state" not in restored:
        raise ValueError("Checkpoint does not contain 'actor_state'.")

    actor_state = restored["actor_state"]
    if hasattr(actor_state, "params"):
        variables = actor_state.params
    elif isinstance(actor_state, Mapping) and "params" in actor_state:
        variables = actor_state["params"]
    else:
        raise ValueError("Unable to read actor params from checkpoint.")

    variables = _to_plain_dict(variables)
    if "params" in variables and isinstance(variables["params"], Mapping):
        return variables["params"]
    return variables


def _resolve_lidar_args(args):
    if args.lidar_profile == "hokuyo":
        if args.obs_dim is None:
            args.obs_dim = 1080
        if args.scan_points is None:
            args.scan_points = 1080
        if args.max_lidar_range is None:
            args.max_lidar_range = 30.0
        if args.lidar_fov_rad is None:
            args.lidar_fov_rad = 4.7
    elif args.lidar_profile == "t_mini_plus":
        if args.obs_dim is None:
            args.obs_dim = 320
        if args.scan_points is None:
            args.scan_points = 320
        if args.max_lidar_range is None:
            args.max_lidar_range = 12.0
        if args.lidar_fov_rad is None:
            args.lidar_fov_rad = 4.7
    else:
        if args.obs_dim is None:
            args.obs_dim = 1080
        if args.max_lidar_range is None:
            args.max_lidar_range = 12.0

    args.obs_dim = int(args.obs_dim)
    args.action_dim = int(args.action_dim)
    if args.scan_points is not None:
        args.scan_points = int(args.scan_points)
    args.max_lidar_range = float(args.max_lidar_range)


def _add_init(initializers, numpy_helper, name: str, arr):
    initializers.append(numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name=name))


def _make_preprocessed_input(
    nodes,
    initializers,
    helper,
    numpy_helper,
    tensor_proto,
    *,
    input_name: str,
    input_layout: str,
    obs_dim: int,
    scan_points: int | None,
    normalize_input: bool,
    max_lidar_range: float,
):
    if input_layout == "flat":
        graph_inputs = [helper.make_tensor_value_info(input_name, tensor_proto.FLOAT, ["batch", int(obs_dim)])]
        x = input_name
        input_desc = f"{input_name}[batch,{obs_dim}]"
    else:
        points = int(obs_dim if scan_points is None else scan_points)
        if points <= 0:
            raise ValueError(f"Invalid scan points: {points}")

        graph_inputs = [helper.make_tensor_value_info(input_name, tensor_proto.FLOAT, ["batch", 1, points])]
        x = input_name
        input_desc = f"{input_name}[batch,1,{points}]"

        shape_name = "scan_flatten_shape"
        initializers.append(numpy_helper.from_array(np.asarray([-1, points], dtype=np.int64), name=shape_name))
        x_flat = "scan_flat"
        nodes.append(
            helper.make_node(
                "Reshape",
                inputs=[x, shape_name],
                outputs=[x_flat],
                name="ScanReshapeToFlat",
            )
        )
        x = x_flat

        if points != int(obs_dim):
            if points < int(obs_dim):
                raise ValueError(
                    f"scan_points ({points}) must be >= obs_dim ({obs_dim}). "
                    "Upsampling is not supported in this exporter."
                )
            idx = np.linspace(0, points - 1, int(obs_dim)).round().astype(np.int64)
            idx_name = "scan_downsample_indices"
            initializers.append(numpy_helper.from_array(idx, name=idx_name))
            x_ds = "scan_downsampled"
            nodes.append(
                helper.make_node(
                    "Gather",
                    inputs=[x, idx_name],
                    outputs=[x_ds],
                    axis=1,
                    name="ScanDownsample",
                )
            )
            x = x_ds
            input_desc += f" -> downsample({points}->{obs_dim})"

    if normalize_input:
        min_name = "norm_min"
        max_name = "norm_max"
        div_name = "norm_div"
        _add_init(initializers, numpy_helper, min_name, np.array(0.0, dtype=np.float32))
        _add_init(initializers, numpy_helper, max_name, np.array(float(max_lidar_range), dtype=np.float32))
        _add_init(initializers, numpy_helper, div_name, np.array(float(max_lidar_range), dtype=np.float32))

        x_clip = "obs_clip"
        nodes.append(
            helper.make_node(
                "Clip",
                inputs=[x, min_name, max_name],
                outputs=[x_clip],
                name="InputClip",
            )
        )
        x_norm = "obs_norm"
        nodes.append(
            helper.make_node(
                "Div",
                inputs=[x_clip, div_name],
                outputs=[x_norm],
                name="InputNormalize",
            )
        )
        x = x_norm
        input_desc += f" + clip/div(max={max_lidar_range})"

    return x, graph_inputs, input_desc


def _add_conv_relu(nodes, initializers, helper, numpy_helper, x, layer, stride: int, name: str):
    # Flax Conv1D kernel: [kernel, in_channel, out_channel]
    # ONNX Conv1D kernel: [out_channel, in_channel, kernel]
    w = _as_numpy(layer["kernel"])
    b = _as_numpy(layer["bias"])
    w = np.transpose(w, (2, 1, 0))

    w_name = f"{name}_W"
    b_name = f"{name}_b"
    _add_init(initializers, numpy_helper, w_name, w)
    _add_init(initializers, numpy_helper, b_name, b)

    y = f"{name}_conv"
    nodes.append(
        helper.make_node(
            "Conv",
            inputs=[x, w_name, b_name],
            outputs=[y],
            strides=[int(stride)],
            auto_pad="SAME_UPPER",
            name=f"{name}_Conv",
        )
    )
    z = f"{name}_relu"
    nodes.append(helper.make_node("Relu", inputs=[y], outputs=[z], name=f"{name}_Relu"))
    return z


def _add_dense(nodes, initializers, helper, numpy_helper, x, layer, name: str, activation: str | None):
    # Flax Dense kernel: [in_features, out_features]
    w = _as_numpy(layer["kernel"])
    b = _as_numpy(layer["bias"])

    w_name = f"{name}_W"
    b_name = f"{name}_b"
    _add_init(initializers, numpy_helper, w_name, w)
    _add_init(initializers, numpy_helper, b_name, b)

    y = f"{name}_gemm"
    nodes.append(
        helper.make_node(
            "Gemm",
            inputs=[x, w_name, b_name],
            outputs=[y],
            alpha=1.0,
            beta=1.0,
            transB=0,
            name=f"{name}_Gemm",
        )
    )

    if activation is None:
        return y
    if activation == "relu":
        z = f"{name}_relu"
        nodes.append(helper.make_node("Relu", inputs=[y], outputs=[z], name=f"{name}_Relu"))
        return z
    if activation == "tanh":
        z = f"{name}_tanh"
        nodes.append(helper.make_node("Tanh", inputs=[y], outputs=[z], name=f"{name}_Tanh"))
        return z
    raise ValueError(f"Unsupported activation: {activation}")


def _compute_encoder_flatten_dim(obs_dim: int, encoder_params):
    conv_keys = _sorted_layer_keys(encoder_params, "Conv")
    if len(conv_keys) < 5:
        raise ValueError(f"Expected at least 5 conv layers in encoder, got {len(conv_keys)}: {conv_keys}")

    conv_keys = conv_keys[:5]
    conv_strides = [4, 4, 2, 1, 1]
    length = int(obs_dim)
    for s in conv_strides:
        length = (length + int(s) - 1) // int(s)  # SAME padding output length = ceil(length / stride)

    last_conv_key = conv_keys[-1]
    last_conv_kernel = _as_numpy(encoder_params[last_conv_key]["kernel"])
    out_channels = int(last_conv_kernel.shape[-1])  # Flax Conv1D kernel: [kernel, in_channel, out_channel]
    return int(length * out_channels)


def _assert_encoder_dense_compat(agent: str, obs_dim: int, encoder_params, first_dense_params, args):
    encoder_flat_dim = _compute_encoder_flatten_dim(int(obs_dim), encoder_params)
    dense_in_dim = int(_as_numpy(first_dense_params["kernel"]).shape[0])  # [in_features, out_features]

    if encoder_flat_dim == dense_in_dim:
        return

    msg = [
        f"{agent.upper()} dimension mismatch:",
        f"  encoder output dim from --obs-dim={int(obs_dim)} is {encoder_flat_dim}",
        f"  but first Dense expects {dense_in_dim} (from checkpoint weights).",
        "This usually means checkpoint training obs_dim and export obs_dim differ.",
    ]

    if int(obs_dim) == 1080 and dense_in_dim == 640:
        msg.append("Hint: this checkpoint likely used obs_dim=320.")
        if getattr(args, "input_layout", "flat") == "scan":
            scan_points = args.scan_points if args.scan_points is not None else int(obs_dim)
            msg.append(
                "Use e.g. --obs-dim 320 "
                f"--scan-points {int(scan_points)} to keep input shape and downsample in ONNX."
            )
        else:
            msg.append("Use e.g. --obs-dim 320 for flat input export.")

    raise ValueError("\n".join(msg))


def _build_encoder(nodes, initializers, helper, numpy_helper, input_name: str, obs_dim: int, encoder_params):
    # [N, obs_dim] -> [N, 1, obs_dim]
    reshape_shape_name = "encoder_input_shape"
    initializers.append(
        numpy_helper.from_array(np.asarray([-1, 1, int(obs_dim)], dtype=np.int64), name=reshape_shape_name)
    )
    x = "encoder_input_ncl"
    nodes.append(
        helper.make_node(
            "Reshape",
            inputs=[input_name, reshape_shape_name],
            outputs=[x],
            name="EncoderInputReshape",
        )
    )

    conv_keys = _sorted_layer_keys(encoder_params, "Conv")
    if len(conv_keys) < 5:
        raise ValueError(f"Expected at least 5 conv layers in encoder, got {len(conv_keys)}: {conv_keys}")
    conv_keys = conv_keys[:5]
    conv_strides = [4, 4, 2, 1, 1]

    for i, (k, s) in enumerate(zip(conv_keys, conv_strides)):
        x = _add_conv_relu(
            nodes,
            initializers,
            helper,
            numpy_helper,
            x,
            encoder_params[k],
            stride=s,
            name=f"encoder_conv{i}",
        )

    # Convert back to Flax layout [N, L, C] before flatten to keep feature order identical.
    x_nlc = "encoder_output_nlc"
    nodes.append(helper.make_node("Transpose", inputs=[x], outputs=[x_nlc], perm=[0, 2, 1], name="EncoderToNLC"))

    out = "encoder_flat"
    nodes.append(helper.make_node("Flatten", inputs=[x_nlc], outputs=[out], axis=1, name="EncoderFlatten"))
    return out


def _build_ppo_onnx(param_tree, obs_dim: int, action_dim: int, opset: int, args):
    import onnx
    from onnx import helper, numpy_helper, TensorProto

    if "encoder" not in param_tree or "actor_mlp" not in param_tree:
        raise ValueError("PPO actor param tree must contain 'encoder' and 'actor_mlp'.")

    nodes = []
    initializers = []

    model_obs, graph_inputs, input_desc = _make_preprocessed_input(
        nodes,
        initializers,
        helper,
        numpy_helper,
        TensorProto,
        input_name=args.input_name,
        input_layout=args.input_layout,
        obs_dim=obs_dim,
        scan_points=args.scan_points,
        normalize_input=bool(args.normalize_input),
        max_lidar_range=float(args.max_lidar_range),
    )

    x = _build_encoder(
        nodes,
        initializers,
        helper,
        numpy_helper,
        input_name=model_obs,
        obs_dim=obs_dim,
        encoder_params=param_tree["encoder"],
    )

    mlp = param_tree["actor_mlp"]
    dense_keys = _sorted_layer_keys(mlp, "Dense")
    if len(dense_keys) < 4:
        raise ValueError(f"Expected 4 dense layers in PPO actor_mlp, got {dense_keys}")
    d0, d1, d2, d3 = dense_keys[:4]

    _assert_encoder_dense_compat("ppo", obs_dim, param_tree["encoder"], mlp[d0], args)

    x = _add_dense(nodes, initializers, helper, numpy_helper, x, mlp[d0], "actor_dense0", "relu")
    x = _add_dense(nodes, initializers, helper, numpy_helper, x, mlp[d1], "actor_dense1", "relu")
    x = _add_dense(nodes, initializers, helper, numpy_helper, x, mlp[d2], "actor_dense2", "relu")
    action_raw = _add_dense(nodes, initializers, helper, numpy_helper, x, mlp[d3], "actor_out", "tanh")

    action_name = args.output_name
    if action_name != action_raw:
        nodes.append(helper.make_node("Identity", inputs=[action_raw], outputs=[action_name], name="ActionRename"))

    graph = helper.make_graph(
        nodes=nodes,
        name="magp_rl_ppo_actor",
        inputs=graph_inputs,
        outputs=[helper.make_tensor_value_info(action_name, TensorProto.FLOAT, ["batch", int(action_dim)])],
        initializer=initializers,
    )

    model = helper.make_model(
        graph,
        producer_name="magp_rl.export_onnx",
        opset_imports=[helper.make_opsetid("", int(opset))],
    )
    onnx.checker.check_model(model)
    return model, input_desc


def _build_sac_onnx(param_tree, obs_dim: int, action_dim: int, opset: int, args):
    import onnx
    from onnx import helper, numpy_helper, TensorProto

    if "encoder" not in param_tree:
        raise ValueError("SAC actor param tree must contain 'encoder'.")

    dense_keys = _sorted_layer_keys(param_tree, "Dense")
    if len(dense_keys) < 4:
        raise ValueError(f"Expected 4 dense layers in SAC actor, got {dense_keys}")
    d0, d1, d2, d3 = dense_keys[:4]

    _assert_encoder_dense_compat("sac", obs_dim, param_tree["encoder"], param_tree[d0], args)

    nodes = []
    initializers = []

    model_obs, graph_inputs, input_desc = _make_preprocessed_input(
        nodes,
        initializers,
        helper,
        numpy_helper,
        TensorProto,
        input_name=args.input_name,
        input_layout=args.input_layout,
        obs_dim=obs_dim,
        scan_points=args.scan_points,
        normalize_input=bool(args.normalize_input),
        max_lidar_range=float(args.max_lidar_range),
    )

    x = _build_encoder(
        nodes,
        initializers,
        helper,
        numpy_helper,
        input_name=model_obs,
        obs_dim=obs_dim,
        encoder_params=param_tree["encoder"],
    )
    x = _add_dense(nodes, initializers, helper, numpy_helper, x, param_tree[d0], "sac_dense0", "relu")
    x = _add_dense(nodes, initializers, helper, numpy_helper, x, param_tree[d1], "sac_dense1", "relu")

    mean_raw = _add_dense(nodes, initializers, helper, numpy_helper, x, param_tree[d2], "sac_mean", None)
    log_std_pre = _add_dense(nodes, initializers, helper, numpy_helper, x, param_tree[d3], "sac_logstd", None)

    _add_init(initializers, numpy_helper, "logstd_min", np.array(-5.0, dtype=np.float32))
    _add_init(initializers, numpy_helper, "logstd_max", np.array(2.0, dtype=np.float32))
    log_std_raw = "sac_logstd_clipped"
    nodes.append(
        helper.make_node(
            "Clip",
            inputs=[log_std_pre, "logstd_min", "logstd_max"],
            outputs=[log_std_raw],
            name="SACLogStdClip",
        )
    )

    action_raw = "sac_action"
    nodes.append(helper.make_node("Tanh", inputs=[mean_raw], outputs=[action_raw], name="SACDeterministicAction"))

    action_name = args.output_name
    mean_name = args.mean_output_name
    logstd_name = args.logstd_output_name

    if action_name != action_raw:
        nodes.append(helper.make_node("Identity", inputs=[action_raw], outputs=[action_name], name="ActionRename"))
    if mean_name != mean_raw:
        nodes.append(helper.make_node("Identity", inputs=[mean_raw], outputs=[mean_name], name="MeanRename"))
    if logstd_name != log_std_raw:
        nodes.append(helper.make_node("Identity", inputs=[log_std_raw], outputs=[logstd_name], name="LogStdRename"))

    outputs = []
    if args.sac_output in ("deterministic", "all"):
        outputs.append(helper.make_tensor_value_info(action_name, TensorProto.FLOAT, ["batch", int(action_dim)]))
    if args.sac_output in ("mean_logstd", "all"):
        outputs.append(helper.make_tensor_value_info(mean_name, TensorProto.FLOAT, ["batch", int(action_dim)]))
        outputs.append(helper.make_tensor_value_info(logstd_name, TensorProto.FLOAT, ["batch", int(action_dim)]))

    graph = helper.make_graph(
        nodes=nodes,
        name="magp_rl_sac_actor",
        inputs=graph_inputs,
        outputs=outputs,
        initializer=initializers,
    )

    model = helper.make_model(
        graph,
        producer_name="magp_rl.export_onnx",
        opset_imports=[helper.make_opsetid("", int(opset))],
    )
    onnx.checker.check_model(model)
    return model, input_desc


def _build_td3_onnx(param_tree, obs_dim: int, action_dim: int, opset: int, args):
    import onnx
    from onnx import helper, numpy_helper, TensorProto

    if "encoder" not in param_tree:
        raise ValueError("TD3 actor param tree must contain 'encoder'.")

    dense_keys = _sorted_layer_keys(param_tree, "Dense")
    if len(dense_keys) < 3:
        raise ValueError(f"Expected 3 dense layers in TD3 actor, got {dense_keys}")
    d0, d1, d2 = dense_keys[:3]

    _assert_encoder_dense_compat("td3", obs_dim, param_tree["encoder"], param_tree[d0], args)

    nodes = []
    initializers = []

    model_obs, graph_inputs, input_desc = _make_preprocessed_input(
        nodes,
        initializers,
        helper,
        numpy_helper,
        TensorProto,
        input_name=args.input_name,
        input_layout=args.input_layout,
        obs_dim=obs_dim,
        scan_points=args.scan_points,
        normalize_input=bool(args.normalize_input),
        max_lidar_range=float(args.max_lidar_range),
    )

    x = _build_encoder(
        nodes,
        initializers,
        helper,
        numpy_helper,
        input_name=model_obs,
        obs_dim=obs_dim,
        encoder_params=param_tree["encoder"],
    )
    x = _add_dense(nodes, initializers, helper, numpy_helper, x, param_tree[d0], "td3_dense0", "relu")
    x = _add_dense(nodes, initializers, helper, numpy_helper, x, param_tree[d1], "td3_dense1", "relu")
    action_raw = _add_dense(nodes, initializers, helper, numpy_helper, x, param_tree[d2], "td3_out", "tanh")

    action_name = args.output_name
    if action_name != action_raw:
        nodes.append(helper.make_node("Identity", inputs=[action_raw], outputs=[action_name], name="TD3ActionRename"))

    graph = helper.make_graph(
        nodes=nodes,
        name="magp_rl_td3_actor",
        inputs=graph_inputs,
        outputs=[helper.make_tensor_value_info(action_name, TensorProto.FLOAT, ["batch", int(action_dim)])],
        initializer=initializers,
    )

    model = helper.make_model(
        graph,
        producer_name="magp_rl.export_onnx",
        opset_imports=[helper.make_opsetid("", int(opset))],
    )
    onnx.checker.check_model(model)
    return model, input_desc


def _make_restore_target(agent: str, obs_dim: int, action_dim: int):
    import jax
    from src.agents.ppo import create_train_states
    from src.agents.sac import create_sac_states
    from src.agents.td3 import create_td3_states

    rng = jax.random.PRNGKey(0)
    obs_shape = (int(obs_dim),)

    if agent == "ppo":
        actor_state, critic_state = create_train_states(
            rng,
            obs_shape=obs_shape,
            action_dim=int(action_dim),
            actor_lr=3e-4,
            critic_lr=1e-3,
        )
        return {"actor_state": actor_state, "critic_state": critic_state, "update": 0}

    if agent == "sac":
        actor_state, critic1_state, critic2_state, target_critic1_params, target_critic2_params, alpha_state = (
            create_sac_states(
                rng,
                obs_shape=obs_shape,
                action_dim=int(action_dim),
                actor_lr=1e-4,
                critic_lr=1e-4,
                alpha_lr=1e-4,
                init_temperature=0.1,
            )
        )
        return {
            "actor_state": actor_state,
            "critic1_state": critic1_state,
            "critic2_state": critic2_state,
            "target_critic1_params": target_critic1_params,
            "target_critic2_params": target_critic2_params,
            "alpha_state": alpha_state,
            "global_step": 0,
        }

    (
        actor_state,
        critic1_state,
        critic2_state,
        target_actor_params,
        target_critic1_params,
        target_critic2_params,
    ) = create_td3_states(
        rng,
        obs_shape=obs_shape,
        action_dim=int(action_dim),
        actor_lr=1e-4,
        critic_lr=1e-4,
    )
    return {
        "actor_state": actor_state,
        "critic1_state": critic1_state,
        "critic2_state": critic2_state,
        "target_actor_params": target_actor_params,
        "target_critic1_params": target_critic1_params,
        "target_critic2_params": target_critic2_params,
        "global_step": 0,
        "update_step": 0,
    }


def _verify_with_onnxruntime(model_path: Path, args):
    try:
        import onnxruntime as ort
    except Exception:
        print("onnxruntime not found. Skip verification.")
        return

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    if args.input_layout == "scan":
        pts = int(args.obs_dim if args.scan_points is None else args.scan_points)
        x = np.random.randn(2, 1, pts).astype(np.float32)
    else:
        x = np.random.randn(2, int(args.obs_dim)).astype(np.float32)
    outs = sess.run(None, {args.input_name: x})
    out_shapes = [tuple(o.shape) for o in outs]
    print(f"ONNXRuntime check passed. output_shapes={out_shapes}")


def main():
    _import_runtime_deps()
    args = _parse_args()
    _resolve_lidar_args(args)

    import jax
    import onnx
    from flax.training import checkpoints

    ckpt_dir, step = _resolve_ckpt_dir_and_step(args.checkpoint_dir, args.step)
    if not ckpt_dir.exists():
        raise SystemExit(f"Checkpoint directory not found: {ckpt_dir}")

    if args.output is None:
        step_tag = f"_step{step}" if step is not None else "_latest"
        output_path = ckpt_dir / f"{args.agent}_actor{step_tag}.onnx"
    else:
        output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    restore_target = _make_restore_target(args.agent, args.obs_dim, args.action_dim)
    restored = checkpoints.restore_checkpoint(
        ckpt_dir=str(ckpt_dir),
        target=restore_target,
        step=step,
    )
    actor_param_tree = _extract_actor_param_tree(restored)

    if args.agent == "ppo":
        model, input_desc = _build_ppo_onnx(actor_param_tree, args.obs_dim, args.action_dim, args.opset, args)
        step_key = "update"
    elif args.agent == "sac":
        model, input_desc = _build_sac_onnx(actor_param_tree, args.obs_dim, args.action_dim, args.opset, args)
        step_key = "global_step"
    else:
        model, input_desc = _build_td3_onnx(actor_param_tree, args.obs_dim, args.action_dim, args.opset, args)
        step_key = "global_step"

    onnx.save(model, str(output_path))
    restored_step = int(np.asarray(jax.device_get(restored.get(step_key, -1))))
    print(f"Exported ONNX: {output_path}")
    print(f"Agent: {args.agent} | Restored {step_key}: {restored_step}")
    print(
        f"Lidar profile: {args.lidar_profile} | obs_dim={args.obs_dim} "
        f"| scan_points={args.scan_points if args.scan_points is not None else args.obs_dim} "
        f"| max_range={args.max_lidar_range} | fov_rad={args.lidar_fov_rad if args.lidar_fov_rad is not None else 'n/a'}"
    )
    print(f"Input: {input_desc}")
    print(f"Outputs: {[o.name for o in model.graph.output]}")

    if args.verify:
        _verify_with_onnxruntime(output_path, args)


if __name__ == "__main__":
    main()
