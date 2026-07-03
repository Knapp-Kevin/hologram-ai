use crate::builder::GraphBuilder;
use anyhow::{anyhow, Result};
use hologram_ai_common::ir::{dtype::DType, graph::AiGraph};
use safetensors::{Dtype as SafeDtype, SafeTensors};
use serde_json::Value;

#[allow(dead_code)]
fn map_dtype(d: SafeDtype) -> Result<DType> {
    match d {
        SafeDtype::F32 => Ok(DType::F32),
        SafeDtype::F16 => Ok(DType::F16),
        SafeDtype::I64 => Ok(DType::INT64),
        SafeDtype::I32 => Ok(DType::INT32),
        _ => Err(anyhow!("Unsupported safetensors dtype: {:?}", d)),
    }
}

pub fn build_parametric_graph(config: &Value, safetensors_shards: &[&[u8]]) -> Result<AiGraph> {
    let mut st_instances = Vec::new();
    for shard in safetensors_shards {
        let st = SafeTensors::deserialize(shard)?;
        st_instances.push(st);
    }

    let mut keys = Vec::new();
    for st in &st_instances {
        for (k, _) in st.tensors() {
            keys.push(k.clone());
        }
    }

    let mut graph = build_parametric_graph_from_keys(config, &keys)?;

    // Inject the actual safetensors weights into the graph's params.
    let mut name_to_id = std::collections::HashMap::new();
    for (id, name) in &graph.tensor_names {
        name_to_id.insert(name.clone(), *id);
    }

    let mut next_id = graph.tensor_names.keys().max().copied().unwrap_or(0) + 1;
    for st in &st_instances {
        for (k, tensor_view) in st.tensors() {
            let id = if let Some(existing_id) = name_to_id.get(&k) {
                *existing_id
            } else {
                let new_id = next_id;
                next_id += 1;
                graph.tensor_names.insert(new_id, k.clone());
                new_id
            };

            let dtype = map_dtype(tensor_view.dtype())?;
            let shape = hologram_ai_common::shape_from_concrete(
                &tensor_view
                    .shape()
                    .iter()
                    .map(|&x| x as u64)
                    .collect::<Vec<_>>(),
            );
            let info = hologram_ai_common::TensorInfo::new(dtype, shape);
            graph.tensor_info.insert(id, info.clone());

            let data = tensor_view.data().to_vec();
            graph.params.insert(
                id,
                hologram_ai_common::ir::param::AiParam::inline(data, info),
            );
        }
    }

    Ok(graph)
}

pub fn build_parametric_graph_from_keys(config: &Value, keys: &[String]) -> Result<AiGraph> {
    let mut builder = GraphBuilder::new("parametric_model".to_string());

    let mut num_layers = 0;
    for key in keys {
        if let Some(idx) = extract_layer_idx(key) {
            if idx >= num_layers {
                num_layers = idx + 1;
            }
        }
    }

    if num_layers == 0 {
        return Err(anyhow!(
            "Could not infer any layers from tensor keys. Is this a transformer?"
        ));
    }

    // Config defaults
    let hidden_size = config
        .get("hidden_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(4096) as u32;
    let num_heads = config
        .get("num_attention_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(32) as u32;
    let _num_kv_heads = config
        .get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(num_heads as u64) as u32;
    let _head_dim = hidden_size / num_heads;

    // Inputs
    let batch = builder.register_var("batch");
    let seq = builder.register_var("seq");
    let input_ids = builder.add_input("input_ids", DType::INT64, vec![batch.clone(), seq.clone()]);

    let vocab_size = config
        .get("vocab_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(32000) as u32;

    let vocab = hologram_ai_common::ir::shape::DimExpr::Concrete(vocab_size as u64);
    let hidden = hologram_ai_common::ir::shape::DimExpr::Concrete(hidden_size as u64);
    let n_heads_expr = hologram_ai_common::ir::shape::DimExpr::Concrete(num_heads as u64);
    let n_kv_heads_expr = hologram_ai_common::ir::shape::DimExpr::Concrete(_num_kv_heads as u64);
    let head_dim_expr = hologram_ai_common::ir::shape::DimExpr::Concrete(_head_dim as u64);

    // 1. Embedding
    let embed_weight = builder.add_tensor(
        "model.embed_tokens.weight",
        DType::F32,
        vec![vocab.clone(), hidden.clone()],
    );
    let mut current = builder.add_tensor(
        "hidden_states",
        DType::F32,
        vec![batch.clone(), seq.clone(), hidden.clone()],
    );
    builder.add_node(
        hologram_ai_common::ir::op::AiOp::Gather { axis: 0 },
        vec![embed_weight, input_ids],
        vec![current],
    );

    // 2. Transformer blocks
    for l in 0..num_layers {
        // Attention Norm
        let attn_norm_weight = builder.add_tensor(
            &format!("model.layers.{l}.input_layernorm.weight"),
            DType::F32,
            vec![hidden.clone()],
        );
        let attn_norm_out = builder.add_tensor(
            &format!("attn_norm_{l}"),
            DType::F32,
            vec![batch.clone(), seq.clone(), hidden.clone()],
        );
        builder.add_node(
            hologram_ai_common::ir::op::AiOp::RmsNorm { epsilon: 1e-5 },
            vec![current, attn_norm_weight],
            vec![attn_norm_out],
        );

        // QKV Projection
        let q_out_var =
            hologram_ai_common::ir::shape::DimExpr::Concrete((num_heads * _head_dim) as u64);
        let k_out_var =
            hologram_ai_common::ir::shape::DimExpr::Concrete((_num_kv_heads * _head_dim) as u64);
        let v_out_var =
            hologram_ai_common::ir::shape::DimExpr::Concrete((_num_kv_heads * _head_dim) as u64);

        let q_proj = builder.add_tensor(
            &format!("model.layers.{l}.self_attn.q_proj.weight"),
            DType::F32,
            vec![hidden.clone(), q_out_var.clone()],
        );
        let k_proj = builder.add_tensor(
            &format!("model.layers.{l}.self_attn.k_proj.weight"),
            DType::F32,
            vec![hidden.clone(), k_out_var.clone()],
        );
        let v_proj = builder.add_tensor(
            &format!("model.layers.{l}.self_attn.v_proj.weight"),
            DType::F32,
            vec![hidden.clone(), v_out_var.clone()],
        );

        let q_out = builder.add_tensor(
            &format!("q_{l}"),
            DType::F32,
            vec![
                batch.clone(),
                seq.clone(),
                n_heads_expr.clone(),
                head_dim_expr.clone(),
            ],
        );
        let k_out = builder.add_tensor(
            &format!("k_{l}"),
            DType::F32,
            vec![
                batch.clone(),
                seq.clone(),
                n_kv_heads_expr.clone(),
                head_dim_expr.clone(),
            ],
        );
        let v_out = builder.add_tensor(
            &format!("v_{l}"),
            DType::F32,
            vec![
                batch.clone(),
                seq.clone(),
                n_kv_heads_expr.clone(),
                head_dim_expr.clone(),
            ],
        );

        builder.add_node(
            hologram_ai_common::ir::op::AiOp::MatMul,
            vec![attn_norm_out, q_proj],
            vec![q_out],
        );
        builder.add_node(
            hologram_ai_common::ir::op::AiOp::MatMul,
            vec![attn_norm_out, k_proj],
            vec![k_out],
        );
        builder.add_node(
            hologram_ai_common::ir::op::AiOp::MatMul,
            vec![attn_norm_out, v_proj],
            vec![v_out],
        );

        // GQA
        let attn_out = builder.add_tensor(
            &format!("attn_out_{l}"),
            DType::F32,
            vec![batch.clone(), seq.clone(), hidden.clone()],
        );
        builder.add_node(
            hologram_ai_common::ir::op::AiOp::GroupedQueryAttention {
                num_heads,
                num_kv_heads: _num_kv_heads,
                head_dim: _head_dim,
                scale: None,
                causal: true,
                heads_first: false,
                qk_norm: false,
                rope: true,
                rope_base: 10000.0,
            },
            vec![q_out, k_out, v_out],
            vec![attn_out],
        );

        // O Projection
        let o_proj = builder.add_tensor(
            &format!("model.layers.{l}.self_attn.o_proj.weight"),
            DType::F32,
            vec![q_out_var.clone(), hidden.clone()],
        );
        let o_out = builder.add_tensor(
            &format!("o_out_{l}"),
            DType::F32,
            vec![batch.clone(), seq.clone(), hidden.clone()],
        );
        builder.add_node(
            hologram_ai_common::ir::op::AiOp::MatMul,
            vec![attn_out, o_proj],
            vec![o_out],
        );

        // Add (residual 1)
        let res1_out = builder.add_tensor(
            &format!("res1_{l}"),
            DType::F32,
            vec![batch.clone(), seq.clone(), hidden.clone()],
        );
        builder.add_node(
            hologram_ai_common::ir::op::AiOp::Add,
            vec![current, o_out],
            vec![res1_out],
        );

        // MLP Norm
        let mlp_norm_weight = builder.add_tensor(
            &format!("model.layers.{l}.post_attention_layernorm.weight"),
            DType::F32,
            vec![hidden.clone()],
        );
        let mlp_norm_out = builder.add_tensor(
            &format!("mlp_norm_{l}"),
            DType::F32,
            vec![batch.clone(), seq.clone(), hidden.clone()],
        );
        builder.add_node(
            hologram_ai_common::ir::op::AiOp::RmsNorm { epsilon: 1e-5 },
            vec![res1_out, mlp_norm_weight],
            vec![mlp_norm_out],
        );

        // MLP Gate + Up
        let intermediate_size = config
            .get("intermediate_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(hidden_size as u64 * 4);
        let ffn_hidden = hologram_ai_common::ir::shape::DimExpr::Concrete(intermediate_size);
        let gate_proj = builder.add_tensor(
            &format!("model.layers.{l}.mlp.gate_proj.weight"),
            DType::F32,
            vec![hidden.clone(), ffn_hidden.clone()],
        );
        let up_proj = builder.add_tensor(
            &format!("model.layers.{l}.mlp.up_proj.weight"),
            DType::F32,
            vec![hidden.clone(), ffn_hidden.clone()],
        );

        let gate_out = builder.add_tensor(
            &format!("gate_out_{l}"),
            DType::F32,
            vec![batch.clone(), seq.clone(), ffn_hidden.clone()],
        );
        let up_out = builder.add_tensor(
            &format!("up_out_{l}"),
            DType::F32,
            vec![batch.clone(), seq.clone(), ffn_hidden.clone()],
        );

        builder.add_node(
            hologram_ai_common::ir::op::AiOp::MatMul,
            vec![mlp_norm_out, gate_proj],
            vec![gate_out],
        );
        builder.add_node(
            hologram_ai_common::ir::op::AiOp::MatMul,
            vec![mlp_norm_out, up_proj],
            vec![up_out],
        );

        let silu_out = builder.add_tensor(
            &format!("silu_out_{l}"),
            DType::F32,
            vec![batch.clone(), seq.clone(), ffn_hidden.clone()],
        );
        builder.add_node(
            hologram_ai_common::ir::op::AiOp::Silu,
            vec![gate_out],
            vec![silu_out],
        );

        let mul_out = builder.add_tensor(
            &format!("mul_out_{l}"),
            DType::F32,
            vec![batch.clone(), seq.clone(), ffn_hidden.clone()],
        );
        builder.add_node(
            hologram_ai_common::ir::op::AiOp::Mul,
            vec![silu_out, up_out],
            vec![mul_out],
        );

        // MLP Down
        let down_proj = builder.add_tensor(
            &format!("model.layers.{l}.mlp.down_proj.weight"),
            DType::F32,
            vec![ffn_hidden.clone(), hidden.clone()],
        );
        let down_out = builder.add_tensor(
            &format!("down_out_{l}"),
            DType::F32,
            vec![batch.clone(), seq.clone(), hidden.clone()],
        );
        builder.add_node(
            hologram_ai_common::ir::op::AiOp::MatMul,
            vec![mul_out, down_proj],
            vec![down_out],
        );

        // Add (residual 2)
        let res2_out = builder.add_tensor(
            &format!("res2_{l}"),
            DType::F32,
            vec![batch.clone(), seq.clone(), hidden.clone()],
        );
        builder.add_node(
            hologram_ai_common::ir::op::AiOp::Add,
            vec![res1_out, down_out],
            vec![res2_out],
        );

        current = res2_out;
    }

    // 3. Final Norm
    let norm_weight = builder.add_tensor("model.norm.weight", DType::F32, vec![hidden.clone()]);
    let norm_out = builder.add_tensor(
        "final_norm",
        DType::F32,
        vec![batch.clone(), seq.clone(), hidden.clone()],
    );
    builder.add_node(
        hologram_ai_common::ir::op::AiOp::RmsNorm { epsilon: 1e-5 },
        vec![current, norm_weight],
        vec![norm_out],
    );

    // 4. LM Head
    let lm_head_weight = builder.add_tensor(
        "lm_head.weight",
        DType::F32,
        vec![hidden.clone(), vocab.clone()],
    );
    let logits = builder.add_tensor(
        "logits",
        DType::F32,
        vec![batch.clone(), seq.clone(), vocab.clone()],
    );
    builder.add_node(
        hologram_ai_common::ir::op::AiOp::MatMul,
        vec![norm_out, lm_head_weight],
        vec![logits],
    );

    // Output
    builder.add_output(logits, "logits");

    let mut graph = builder.build();

    graph.metadata.insert(
        "vocab_size".to_string(),
        hologram_ai_common::MetaValue::Int(vocab_size as i64),
    );
    graph.metadata.insert(
        "arch".to_string(),
        hologram_ai_common::MetaValue::Str("parametric_transformer".to_string()),
    );
    graph.metadata.insert(
        "n_layers".to_string(),
        hologram_ai_common::MetaValue::Int(num_layers as i64),
    );

    Ok(graph)
}

fn extract_layer_idx(key: &str) -> Option<usize> {
    let parts: Vec<&str> = key.split('.').collect();
    for (i, part) in parts.iter().enumerate() {
        if (*part == "layers" || *part == "h" || *part == "blocks") && i + 1 < parts.len() {
            if let Ok(idx) = parts[i + 1].parse::<usize>() {
                return Some(idx);
            }
        }
    }
    None
}
