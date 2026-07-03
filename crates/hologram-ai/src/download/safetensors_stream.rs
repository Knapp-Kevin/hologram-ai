use crate::download::hf_api::{HfClient, ModelInfo};
use anyhow::{Context, Result};
use futures::StreamExt;
use hologram_ai_common::ir::{AiParam, DType, Dim, TensorInfo};
use std::collections::HashMap;

// This will stream safetensors and return a mapping of tensor names to AiParam::External.
// It also needs to fetch the config.json.
