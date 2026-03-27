//! Resolve Slice parameters from constant inputs.
//!
//! ONNX opset 10+ encodes Slice axes/starts/ends/steps as input tensors
//! rather than op attributes. The ONNX importer creates placeholder
//! `AiOp::Slice { axes: [], starts: [], ends: [], steps: [] }` nodes.
//!
//! This pass reads the constant param inputs and fills in the AiOp struct
//! fields, enabling `SliceToGather` and the lowering strategy to handle them.

use super::pipeline::Pass;
use crate::ir::{AiGraph, AiOp, AiParam};

pub struct ResolveSliceParams;

impl Pass for ResolveSliceParams {
    fn name(&self) -> &str {
        "ResolveSliceParams"
    }

    fn run(&self, mut graph: AiGraph) -> anyhow::Result<AiGraph> {
        for node in &mut graph.nodes {
            if let AiOp::Slice {
                axes,
                starts,
                ends,
                steps,
            } = &mut node.op
            {
                // Skip if already resolved (attribute-based form).
                if !starts.is_empty() {
                    continue;
                }

                // ONNX Slice inputs: [data, starts, ends, axes?, steps?]
                // Read starts from input[1], ends from input[2],
                // axes from input[3] (optional), steps from input[4] (optional).
                let resolved_starts = read_i64_param(&graph.params, node.inputs.get(1).copied());
                let resolved_ends = read_i64_param(&graph.params, node.inputs.get(2).copied());
                let resolved_axes = read_i64_param(&graph.params, node.inputs.get(3).copied());
                let resolved_steps = read_i64_param(&graph.params, node.inputs.get(4).copied());

                // Resolve what we can. If ends is unknown (dynamic, e.g.,
                // sequence length), use i64::MAX as the ONNX sentinel for
                // "slice to end of axis". This lets the lowering and runtime
                // handle the dynamic bound correctly.
                if let Some(s) = resolved_starts {
                    let n = s.len();
                    let e = resolved_ends.unwrap_or_else(|| vec![i64::MAX; n]);
                    *starts = s;
                    *ends = e;
                    *axes = resolved_axes.unwrap_or_else(|| (0..n as i64).collect());
                    *steps = resolved_steps.unwrap_or_else(|| vec![1; n]);
                }
            }
        }

        Ok(graph)
    }
}

/// Read an i64 vector from a constant parameter tensor.
fn read_i64_param(
    params: &std::collections::HashMap<crate::ir::TensorId, AiParam>,
    tid: Option<crate::ir::TensorId>,
) -> Option<Vec<i64>> {
    let tid = tid?;
    let param = params.get(&tid)?;
    let (data, info) = match param {
        AiParam::Inline { data, info } => (data.as_slice(), info),
        AiParam::Mmap { .. } => return None, // Can't read mmap at compile time easily
    };
    if data.is_empty() {
        return None;
    }
    match info.logical_dtype {
        crate::ir::DType::INT64 => {
            let values: Vec<i64> = data
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().expect("chunk is 8 bytes")))
                .collect();
            Some(values)
        }
        crate::ir::DType::INT32 => {
            let values: Vec<i64> = data
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().expect("chunk is 4 bytes")) as i64)
                .collect();
            Some(values)
        }
        _ => None,
    }
}
