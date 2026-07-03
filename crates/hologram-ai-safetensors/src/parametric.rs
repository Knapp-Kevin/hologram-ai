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

pub fn build_parametric_graph(_config: &Value, safetensors_bytes: &[u8]) -> Result<AiGraph> {
    let builder = GraphBuilder::new("parametric_model".to_string());

    // 1. Identify tensors from safetensors keys
    let st = SafeTensors::deserialize(safetensors_bytes)?;
    let tensors = st.tensors();
    let keys: Vec<&String> = tensors.iter().map(|(k, _)| k).collect();

    // Determine number of layers by looking for the max layer index in keys
    let mut num_layers = 0;
    for key in &keys {
        if let Some(idx) = extract_layer_idx(key) {
            if idx >= num_layers {
                num_layers = idx + 1;
            }
        }
    }

    if num_layers == 0 {
        return Err(anyhow!("Could not infer any layers from tensor keys"));
    }

    // Find embedding tensor (usually contains "embed" or "wte")
    let _embed_key = keys
        .iter()
        .find(|k| k.contains("embed") || k.contains("wte"))
        .ok_or_else(|| anyhow!("Could not find embedding tensor"))?;

    // (Here we would dynamically construct the generic transformer graph...)
    // This removes the hardcoded `match config.model_type` logic.

    Ok(builder.build())
}

fn extract_layer_idx(key: &str) -> Option<usize> {
    // Looks for patterns like "layers.0." or "h.0."
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
