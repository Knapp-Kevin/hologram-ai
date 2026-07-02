//! Shared graph-traversal and mutation helpers for optimization passes.

use crate::ir::{AiGraph, AiNode, TensorId};
use std::collections::{HashMap, HashSet};

/// Map each output tensor to the index of the node that produces it.
pub fn build_producer_map(graph: &AiGraph) -> HashMap<TensorId, usize> {
    graph
        .nodes
        .iter()
        .enumerate()
        .flat_map(|(i, n)| n.outputs.iter().map(move |&tid| (tid, i)))
        .collect()
}

/// Map each tensor to all `(consumer_node_index, input_position)` pairs.
pub fn build_consumer_map(graph: &AiGraph) -> HashMap<TensorId, Vec<(usize, usize)>> {
    let mut consumers: HashMap<TensorId, Vec<(usize, usize)>> = HashMap::new();
    for (i, n) in graph.nodes.iter().enumerate() {
        for (pos, &tid) in n.inputs.iter().enumerate() {
            consumers.entry(tid).or_default().push((i, pos));
        }
    }
    consumers
}

/// True when the tensor has exactly one consumer.
pub fn has_single_consumer(
    tid: TensorId,
    consumers: &HashMap<TensorId, Vec<(usize, usize)>>,
) -> bool {
    consumers.get(&tid).is_some_and(|c| c.len() == 1)
}

/// Next available node ID (one past the current max).
pub fn next_node_id(graph: &AiGraph) -> u32 {
    graph.nodes.iter().map(|n| n.id).max().unwrap_or(0) + 1
}

/// Apply removals and replacements to `graph.nodes`, then invalidate the
/// topological sort cache.
///
/// - Nodes at indices in `to_remove` are dropped.
/// - Nodes at indices in `replacements` are swapped with the replacement.
/// - All other nodes are kept as-is.
///
/// Returns the number of mutations applied (removals + replacements).
pub fn apply_node_mutations(
    graph: &mut AiGraph,
    to_remove: &HashSet<usize>,
    replacements: &mut HashMap<usize, AiNode>,
) -> usize {
    if to_remove.is_empty() && replacements.is_empty() {
        return 0;
    }
    let count = to_remove.len() + replacements.len();
    let new_nodes: Vec<AiNode> = graph
        .nodes
        .drain(..)
        .enumerate()
        .filter_map(|(idx, node)| {
            if let Some(replacement) = replacements.remove(&idx) {
                Some(replacement)
            } else if to_remove.contains(&idx) {
                None
            } else {
                Some(node)
            }
        })
        .collect();
    graph.nodes = new_nodes;
    graph.invalidate_topo_cache();
    count
}

/// Remove nodes at the given indices. Shorthand for `apply_node_mutations`
/// with no replacements.
pub fn remove_nodes(graph: &mut AiGraph, to_remove: &HashSet<usize>) -> usize {
    let mut empty = HashMap::new();
    apply_node_mutations(graph, to_remove, &mut empty)
}

// ── Helpers hoisted from AttentionFusion for use in declarative rules ──

use crate::ir::{AiOp, AiParam};

/// Matched SDPA chain info.
pub struct SdpaChain {
    pub v_tid: TensorId,
    pub scale: f32,
    pub has_mask: bool,
    #[allow(dead_code)]
    pub mask_tid: Option<TensorId>,
    pub output_tid: TensorId,
    pub output_matmul_idx: usize,
    pub node_indices: Vec<usize>,
}

/// Try to match: scores → [Mul(scale)] → [Add(mask)] → Softmax → [IsNaN→Where] → MatMul(_, V).
pub fn match_sdpa_chain(
    scores_tid: TensorId,
    _tid_to_node: &HashMap<TensorId, usize>,
    consumers: &HashMap<TensorId, Vec<(usize, usize)>>,
    graph: &AiGraph,
) -> Option<SdpaChain> {
    let mut current_tid = scores_tid;
    let mut chain_indices: Vec<usize> = Vec::new();
    let mut scale = 1.0f32;
    let mut has_mask = false;

    // Optional: Mul(scores, scale_constant)
    if let Some(next) =
        find_consumer_by_op(current_tid, consumers, graph, |op| matches!(op, AiOp::Mul))
    {
        let n = &graph.nodes[next];
        if matches!(n.op, AiOp::Mul) && n.inputs.len() >= 2 {
            let (other_idx, scale_val) = if let Some(s) = scalar_param(n.inputs[1], graph) {
                (0, s)
            } else if let Some(s) = scalar_param(n.inputs[0], graph) {
                (1, s)
            } else {
                (usize::MAX, 0.0)
            };
            if other_idx < 2 && n.inputs[other_idx] == current_tid {
                scale = scale_val;
                chain_indices.push(next);
                current_tid = *n.outputs.first()?;
            }
        }
    }

    // Optional: Add(scores, mask)
    let mut mask_tid: Option<TensorId> = None;
    if let Some(next) =
        find_consumer_by_op(current_tid, consumers, graph, |op| matches!(op, AiOp::Add))
    {
        let n = &graph.nodes[next];
        if n.inputs.len() >= 2 && (n.inputs[0] == current_tid || n.inputs[1] == current_tid) {
            has_mask = true;
            mask_tid = if n.inputs[0] == current_tid {
                Some(n.inputs[1])
            } else {
                Some(n.inputs[0])
            };
            chain_indices.push(next);
            current_tid = *n.outputs.first()?;
        }
    }

    // Required: Softmax(axis=-1)
    let softmax_consumer = find_consumer_by_op(current_tid, consumers, graph, |op| {
        matches!(op, AiOp::Softmax { .. })
    })?;
    let softmax_node = &graph.nodes[softmax_consumer];
    chain_indices.push(softmax_consumer);
    current_tid = *softmax_node.outputs.first()?;

    // Optional: IsNaN → Where
    if let Some(next) = find_consumer_by_op(current_tid, consumers, graph, |op| {
        matches!(op, AiOp::IsNaN)
    }) {
        let n = &graph.nodes[next];
        if matches!(n.op, AiOp::IsNaN) {
            let isnan_out = *n.outputs.first()?;
            chain_indices.push(next);
            if let Some(where_idx) = single_consumer(isnan_out, consumers) {
                let where_node = &graph.nodes[where_idx];
                if matches!(where_node.op, AiOp::Where) {
                    chain_indices.push(where_idx);
                    current_tid = *where_node.outputs.first()?;
                }
            }
        }
    }

    // Required: MatMul(weights, V)
    let out_matmul = find_consumer_by_op(current_tid, consumers, graph, |op| {
        matches!(op, AiOp::MatMul)
    })?;
    let out_node = &graph.nodes[out_matmul];
    if !matches!(out_node.op, AiOp::MatMul) || out_node.inputs.len() < 2 {
        return None;
    }
    let v_tid = if out_node.inputs[0] == current_tid {
        out_node.inputs[1]
    } else if out_node.inputs[1] == current_tid {
        out_node.inputs[0]
    } else {
        return None;
    };
    let output_tid = *out_node.outputs.first()?;

    Some(SdpaChain {
        v_tid,
        scale,
        has_mask,
        mask_tid,
        output_tid,
        output_matmul_idx: out_matmul,
        node_indices: chain_indices,
    })
}

pub fn find_consumer_by_op(
    tid: TensorId,
    consumers: &HashMap<TensorId, Vec<(usize, usize)>>,
    graph: &AiGraph,
    pred: impl Fn(&AiOp) -> bool,
) -> Option<usize> {
    let c = consumers.get(&tid)?;
    c.iter()
        .map(|&(node_idx, _)| node_idx)
        .find(|&node_idx| pred(&graph.nodes[node_idx].op))
}

pub fn single_consumer(
    tid: TensorId,
    consumers: &HashMap<TensorId, Vec<(usize, usize)>>,
) -> Option<usize> {
    let c = consumers.get(&tid)?;
    if c.len() == 1 {
        Some(c[0].0)
    } else {
        None
    }
}

pub fn scalar_param(tid: TensorId, graph: &AiGraph) -> Option<f32> {
    match graph.params.get(&tid)? {
        AiParam::Inline { data, .. } if data.len() == 4 => {
            let arr: [u8; 4] = data.as_slice().try_into().ok()?;
            Some(f32::from_le_bytes(arr))
        }
        _ => None,
    }
}

pub fn infer_all_head_params(
    q_tid: TensorId,
    k_tid: TensorId,
    _v_tid: TensorId,
    graph: &AiGraph,
) -> (u32, u32, u32) {
    let head_dim = extract_last_dim(q_tid, graph).unwrap_or(0);
    if head_dim == 0 {
        return (0, 0, 0);
    }
    let q_heads = extract_heads_dim(q_tid, graph);
    let k_heads = extract_heads_dim(k_tid, graph);
    let num_heads = q_heads.unwrap_or(0) as u32;
    let num_kv_heads = k_heads.unwrap_or(num_heads as u64) as u32;
    (num_heads, num_kv_heads, head_dim as u32)
}

pub fn extract_last_dim(tid: TensorId, graph: &AiGraph) -> Option<u64> {
    let info = graph.tensor_info.get(&tid)?;
    info.shape.as_slice().last()?.evaluate()
}

pub fn extract_heads_dim(tid: TensorId, graph: &AiGraph) -> Option<u64> {
    let info = graph.tensor_info.get(&tid)?;
    let shape = info.shape.as_slice();
    if shape.len() >= 4 {
        if let Some(h) = shape[1].evaluate() {
            return Some(h);
        }
        return None;
    }
    if shape.len() >= 3 {
        return shape[0].evaluate();
    }
    None
}

pub fn find_pre_transpose_with_scale(
    tid: TensorId,
    tid_to_node: &HashMap<TensorId, usize>,
    graph: &AiGraph,
) -> (TensorId, Option<f32>) {
    let mut current = tid;
    let mut accumulated_scale: Option<f32> = None;
    for _ in 0..5 {
        let node_idx = match tid_to_node.get(&current) {
            Some(&idx) => idx,
            None => return (tid, accumulated_scale),
        };
        let node = &graph.nodes[node_idx];
        match &node.op {
            AiOp::Transpose { .. } => {
                let pre = node.inputs.first().copied().unwrap_or(tid);
                return (pre, accumulated_scale);
            }
            AiOp::Reshape { .. }
            | AiOp::Flatten { .. }
            | AiOp::Squeeze { .. }
            | AiOp::Unsqueeze { .. }
            | AiOp::Expand
            | AiOp::Identity => {
                current = match node.inputs.first() {
                    Some(&inp) => inp,
                    None => return (tid, accumulated_scale),
                };
            }
            AiOp::Mul if node.inputs.len() >= 2 => {
                let s1 = scalar_param(node.inputs[1], graph);
                let s0 = scalar_param(node.inputs[0], graph);
                if let Some(sv) = s1 {
                    accumulated_scale = Some(accumulated_scale.unwrap_or(1.0) * sv);
                    current = node.inputs[0];
                } else if let Some(sv) = s0 {
                    accumulated_scale = Some(accumulated_scale.unwrap_or(1.0) * sv);
                    current = node.inputs[1];
                } else {
                    return (tid, accumulated_scale);
                }
            }
            _ => return (tid, accumulated_scale),
        }
    }
    (tid, accumulated_scale)
}

pub fn trace_past_expand(
    tid: TensorId,
    tid_to_node: &HashMap<TensorId, usize>,
    graph: &AiGraph,
) -> TensorId {
    let mut current = tid;
    for _ in 0..8 {
        let node_idx = match tid_to_node.get(&current) {
            Some(&idx) => idx,
            None => return current,
        };
        let node = &graph.nodes[node_idx];
        match &node.op {
            AiOp::Reshape { .. }
            | AiOp::Flatten { .. }
            | AiOp::Squeeze { .. }
            | AiOp::Unsqueeze { .. }
            | AiOp::Expand
            | AiOp::Identity => {
                current = match node.inputs.first() {
                    Some(&inp) => inp,
                    None => return current,
                };
            }
            _ => return current,
        }
    }
    current
}
