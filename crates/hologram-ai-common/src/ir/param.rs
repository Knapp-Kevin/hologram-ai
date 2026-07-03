use super::graph::TensorInfo;
use std::path::PathBuf;
use std::sync::Arc;

/// A model weight or constant tensor.
///
/// Weights remain in their source representation until the lowering pass
/// decides the dequantization strategy.
///
/// `Inline` data uses `Arc<Vec<u8>>` so that `AiGraph::clone()` is cheap —
/// weight bytes are reference-counted, not deep-copied. This enables
/// parallel compilation of prefill/decode/verify graphs from cloned graphs.
#[derive(Debug, Clone)]
pub enum AiParam {
    /// Small weights embedded directly in the graph.
    Inline {
        data: Arc<Vec<u8>>,
        info: TensorInfo,
    },
    /// Large weights memory-mapped from the source model file.
    Mmap {
        path: PathBuf,
        offset: u64,
        len: u64,
        info: TensorInfo,
    },
    /// External weight stored in holospaces. The runtime will resolve it by Kappa hash.
    External { kappa: String, info: TensorInfo },
}

impl AiParam {
    /// Construct an inline parameter from owned bytes.
    pub fn inline(data: Vec<u8>, info: TensorInfo) -> Self {
        Self::Inline {
            data: Arc::new(data),
            info,
        }
    }

    /// Construct a memory-mapped parameter reference.
    pub fn mmap(path: PathBuf, offset: u64, len: u64, info: TensorInfo) -> Self {
        Self::Mmap {
            path,
            offset,
            len,
            info,
        }
    }

    /// Construct an external parameter reference.
    pub fn external(kappa: String, info: TensorInfo) -> Self {
        Self::External { kappa, info }
    }

    /// Metadata for this parameter.
    pub fn info(&self) -> &TensorInfo {
        match self {
            AiParam::Inline { info, .. } => info,
            AiParam::Mmap { info, .. } => info,
            AiParam::External { info, .. } => info,
        }
    }

    /// Interpret inline data as f32 slice (for small constant params like scales).
    /// Returns `None` for mmap params or if the data isn't aligned/sized for f32.
    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        match self {
            AiParam::Inline { data, .. } if data.len() >= 4 => bytemuck::try_cast_slice(data).ok(),
            _ => None,
        }
    }

    /// Whether this parameter has no backing data (invalid).
    pub fn is_empty(&self) -> bool {
        match self {
            AiParam::Inline { data, .. } => data.is_empty(),
            AiParam::Mmap { len, .. } => *len == 0,
            AiParam::External { .. } => false, // Externals implicitly have data
        }
    }
}
