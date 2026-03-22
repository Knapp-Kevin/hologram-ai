//! Concat + MatMul fusion.
//!
//! Detects the multi-head attention output projection pattern and fuses into
//! a single op, avoiding materializing the concatenated heads buffer.
//!
//! # Pattern
//!
//! ```text
//! concat_out = Concat([h1, h2, ..., hN])
//! proj_out   = MatMul(concat_out, W_out)
//! ```
//!
//! Fused into:
//!
//! ```text
//! proj_out = ConcatMatMul([h1, h2, ..., hN], W_out)
//! ```
//!
//! This pattern appears once per attention layer (multi-head output projection).
//! Fusing eliminates the concatenated heads intermediate buffer.
//!
//! # Prerequisites
//!
//! **Not yet registered in the pipeline** — requires a corresponding
//! `FloatOp::ConcatMatMul` + kernel in hologram base. See Plan 019.

use super::pipeline::Pass;
use crate::ir::{AiGraph, AiNode, AiOp, TensorId};
use std::collections::{HashMap, HashSet};

/// Fuse `Concat + MatMul` into `AiOp::ConcatMatMul`.
pub struct ConcatMatMulFusion;

impl Pass for ConcatMatMulFusion {
    fn name(&self) -> &str {
        "ConcatMatMulFusion"
    }

    fn run(&self, mut graph: AiGraph) -> anyhow::Result<AiGraph> {
        let tid_to_node: HashMap<TensorId, usize> = graph
            .nodes
            .iter()
            .enumerate()
            .flat_map(|(i, n)| n.outputs.iter().map(move |&tid| (tid, i)))
            .collect();

        let mut consumers: HashMap<TensorId, Vec<(usize, usize)>> = HashMap::new();
        for (i, n) in graph.nodes.iter().enumerate() {
            for (pos, &tid) in n.inputs.iter().enumerate() {
                consumers.entry(tid).or_default().push((i, pos));
            }
        }

        let mut to_remove: HashSet<usize> = HashSet::new();
        let mut replacements: HashMap<usize, AiNode> = HashMap::new();
        let mut fused_count: u32 = 0;

        for (mm_idx, mm_node) in graph.nodes.iter().enumerate() {
            if !matches!(mm_node.op, AiOp::MatMul | AiOp::BatchMatMul) {
                continue;
            }
            if mm_node.inputs.len() < 2 || mm_node.outputs.is_empty() {
                continue;
            }
            if to_remove.contains(&mm_idx) {
                continue;
            }

            // Check if the first input (A) comes from a Concat node.
            let a_tid = mm_node.inputs[0];
            let concat_idx = match tid_to_node.get(&a_tid) {
                Some(&idx) => idx,
                None => continue,
            };
            let concat_node = &graph.nodes[concat_idx];
            if !matches!(concat_node.op, AiOp::Concat { .. }) {
                continue;
            }

            // Concat output must have exactly one consumer (this MatMul).
            let concat_out_tid = concat_node.outputs[0];
            let concat_consumers = consumers.get(&concat_out_tid).map_or(0, |c| c.len());
            if concat_consumers != 1 {
                tracing::trace!(
                    concat_idx,
                    concat_consumers,
                    "ConcatMatMulFusion: Concat output has multiple consumers, skipping"
                );
                continue;
            }

            let n_concat_inputs = concat_node.inputs.len() as u32;
            let w_tid = mm_node.inputs[1]; // weight matrix
            let mm_out_tid = mm_node.outputs[0];

            // Build fused inputs: all Concat inputs + weight matrix.
            let mut fused_inputs: Vec<TensorId> = concat_node.inputs.clone();
            fused_inputs.push(w_tid);

            let fused = AiNode::new(
                mm_node.id,
                AiOp::ConcatMatMul { n_concat_inputs },
                fused_inputs,
                vec![mm_out_tid],
            );

            to_remove.insert(concat_idx);
            replacements.insert(mm_idx, fused);
            fused_count += 1;

            tracing::debug!(
                concat_idx,
                mm_idx,
                n_concat_inputs,
                "ConcatMatMulFusion: fused Concat + MatMul → ConcatMatMul"
            );
        }

        if fused_count > 0 {
            tracing::info!(fused_count, "ConcatMatMulFusion: fused pairs");
        }

        let mut new_nodes = Vec::with_capacity(graph.nodes.len() - to_remove.len());
        for (i, node) in graph.nodes.into_iter().enumerate() {
            if to_remove.contains(&i) {
                continue;
            }
            if let Some(fused) = replacements.remove(&i) {
                new_nodes.push(fused);
            } else {
                new_nodes.push(node);
            }
        }
        graph.nodes = new_nodes;
        graph.invalidate_topo_cache();

        Ok(graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{AiGraph, AiNode, AiOp};

    fn empty_graph() -> AiGraph {
        AiGraph {
            name: "test".to_string(),
            nodes: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            input_names: Vec::new(),
            output_names: Vec::new(),
            params: Default::default(),
            tensor_info: Default::default(),
            metadata: Default::default(),
            warnings: Vec::new(),
            dim_vars: Default::default(),
            shape_constraints: Default::default(),
            subgraphs: Default::default(),
            tensor_names: Default::default(),
            topo_cache: Default::default(),
        }
    }

    #[test]
    fn fuses_concat_matmul() {
        let mut g = empty_graph();
        // 4 heads → Concat → MatMul with W_out
        g.inputs = vec![1, 2, 3, 4, 50]; // h1..h4, W_out
        g.outputs = vec![70];
        g.nodes = vec![
            AiNode::new(0, AiOp::Concat { axis: -1 }, vec![1, 2, 3, 4], vec![60]),
            AiNode::new(1, AiOp::MatMul, vec![60, 50], vec![70]),
        ];

        let result = ConcatMatMulFusion.run(g).expect("pass should succeed");
        assert_eq!(result.nodes.len(), 1);
        assert!(matches!(
            result.nodes[0].op,
            AiOp::ConcatMatMul { n_concat_inputs: 4 }
        ));
        assert_eq!(result.nodes[0].inputs, vec![1, 2, 3, 4, 50]);
        assert_eq!(result.nodes[0].outputs, vec![70]);
    }

    #[test]
    fn skips_when_concat_has_multiple_consumers() {
        let mut g = empty_graph();
        g.inputs = vec![1, 2, 50];
        g.outputs = vec![70, 80];
        g.nodes = vec![
            AiNode::new(0, AiOp::Concat { axis: -1 }, vec![1, 2], vec![60]),
            AiNode::new(1, AiOp::MatMul, vec![60, 50], vec![70]),
            AiNode::new(2, AiOp::Relu, vec![60], vec![80]),
        ];

        let result = ConcatMatMulFusion.run(g).expect("pass should succeed");
        assert_eq!(result.nodes.len(), 3);
    }

    #[test]
    fn no_fusion_when_input_is_not_concat() {
        let mut g = empty_graph();
        g.inputs = vec![10, 11];
        g.outputs = vec![30];
        g.nodes = vec![AiNode::new(0, AiOp::MatMul, vec![10, 11], vec![30])];

        let result = ConcatMatMulFusion.run(g).expect("pass should succeed");
        assert_eq!(result.nodes.len(), 1);
        assert!(matches!(result.nodes[0].op, AiOp::MatMul));
    }
}
