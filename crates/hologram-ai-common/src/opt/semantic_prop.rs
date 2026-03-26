//! Semantic hint propagation pass.
//!
//! Infers `SemanticHint` for tensor edges based on the producing op.
//! Runs after fusion passes so that fused ops (GroupedQueryAttention,
//! FusedSwiGLU, etc.) are already present.

use super::pipeline::Pass;
use crate::ir::{AiGraph, AiOp, SemanticHint};

/// Propagate semantic hints through the graph in topological order.
///
/// Rules:
/// - `Embed` → `Embedding`
/// - `Softmax` → `AttentionWeight`
/// - `RmsNorm | LayerNorm | GroupNorm | BatchNorm` → `NormOutput`
/// - `RotaryEmbedding` → `Position`
/// - `GroupedQueryAttention | MultiHeadAttention` → `Residual`
/// - Default: inherit from first input (same as dtype propagation)
pub struct SemanticPropagation;

impl Pass for SemanticPropagation {
    fn name(&self) -> &str {
        "SemanticPropagation"
    }

    fn run(&self, mut graph: AiGraph) -> anyhow::Result<AiGraph> {
        let order = graph.topo_order();

        for &nid in order.iter() {
            let node = match graph.nodes.iter().find(|n| n.id == nid) {
                Some(n) => n,
                None => continue,
            };

            let hint = infer_semantic(&node.op, &node.inputs, &graph);

            for &tid in &node.outputs {
                if let Some(info) = graph.tensor_info.get_mut(&tid) {
                    if info.semantic == SemanticHint::Unknown {
                        info.semantic = hint;
                    }
                }
            }
        }

        Ok(graph)
    }
}

fn infer_semantic(op: &AiOp, inputs: &[crate::ir::TensorId], graph: &AiGraph) -> SemanticHint {
    match op {
        AiOp::Embed => SemanticHint::Embedding,

        AiOp::Softmax { .. } => SemanticHint::AttentionWeight,

        AiOp::RmsNorm { .. }
        | AiOp::LayerNorm { .. }
        | AiOp::GroupNorm { .. }
        | AiOp::BatchNorm { .. }
        | AiOp::FusedLayerNormResidual { .. } => SemanticHint::NormOutput,

        AiOp::RotaryEmbedding { .. } => SemanticHint::Position,

        AiOp::GroupedQueryAttention { .. } | AiOp::MultiHeadAttention { .. } => {
            SemanticHint::Residual
        }

        // Residual add: if either input is Residual, output is Residual
        AiOp::Add => {
            for &tid in inputs {
                if let Some(info) = graph.tensor_info.get(&tid) {
                    if info.semantic == SemanticHint::Residual {
                        return SemanticHint::Residual;
                    }
                }
            }
            inherit_first(inputs, graph)
        }

        // Default: inherit from first input
        _ => inherit_first(inputs, graph),
    }
}

fn inherit_first(inputs: &[crate::ir::TensorId], graph: &AiGraph) -> SemanticHint {
    inputs
        .first()
        .and_then(|tid| graph.tensor_info.get(tid))
        .map(|info| info.semantic)
        .unwrap_or(SemanticHint::Unknown)
}
