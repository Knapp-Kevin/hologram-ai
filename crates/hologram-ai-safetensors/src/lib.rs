pub mod builder;
pub mod llama;


use anyhow::{bail, Context, Result};
use hologram_ai_common::ir::graph::AiGraph;
use serde::Deserialize;

#[derive(Deserialize, Debug)]
#[serde(untagged)]
pub enum ArchConfig {
    Llama(llama::LlamaConfig),
    // Extensible for future architectures
}

#[derive(Deserialize, Debug)]
pub struct ModelConfig {
    pub model_type: String,
    #[serde(flatten)]
    pub arch: ArchConfig,
}

pub fn build_graph_from_safetensors(
    config_json: &str,
    safetensors_bytes: &[u8],
) -> Result<AiGraph> {
    let config: ModelConfig = serde_json::from_str(config_json)
        .context("Failed to parse config.json")?;

    match config.model_type.as_str() {
        "llama" => {
            if let ArchConfig::Llama(llama_config) = config.arch {
                llama::build_llama_graph(llama_config, safetensors_bytes)
            } else {
                bail!("Invalid config for llama");
            }
        }
        other => bail!("Architecture '{}' is not supported yet", other),
    }
}
