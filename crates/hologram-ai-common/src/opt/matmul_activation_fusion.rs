//! MatMul + Activation fusion.
//!
//! Detects MatMul followed by an elementwise activation and fuses into a
//! single op, avoiding the intermediate buffer.
//!
//! # Pattern
//!
//! ```text
//! matmul_out = MatMul(a, b)
//! act_out    = Relu(matmul_out)   — or Gelu, Silu
//! ```
//!
//! Fused into:
//!
//! ```text
//! act_out = MatMulRelu(a, b)      — or MatMulGelu, MatMulSilu
//! ```
//!
//! This pattern appears in FFN output projections and some attention layers.
//! Fusing eliminates the intermediate MatMul output buffer and one dispatch.
//!
//! # Prerequisites
//!
//! **Not yet registered in the pipeline** — requires corresponding fused
//! `FloatOp` variants + kernels in hologram base. See Plan 019.

use super::pipeline::Pass;
use crate::ir::{AiGraph, AiNode, AiOp, TensorId};
use std::collections::{HashMap, HashSet};

/// Fuse `MatMul + Activation` into `AiOp::MatMulRelu`/`MatMulGelu`/`MatMulSilu`.
pub struct MatMulActivationFusion;

impl Pass for MatMulActivationFusion {
    fn name(&self) -> &str {
        "MatMulActivationFusion"
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

        for (act_idx, act_node) in graph.nodes.iter().enumerate() {
            let fused_op = match &act_node.op {
                AiOp::Relu => AiOp::MatMulRelu,
                AiOp::Gelu | AiOp::GeluApprox => AiOp::MatMulGelu,
                AiOp::Silu => AiOp::MatMulSilu,
                _ => continue,
            };

            if act_node.inputs.is_empty() || act_node.outputs.is_empty() {
                continue;
            }
            if to_remove.contains(&act_idx) {
                continue;
            }

            let act_input_tid = act_node.inputs[0];

            // Check if input comes from a MatMul.
            let mm_idx = match tid_to_node.get(&act_input_tid) {
                Some(&idx) => idx,
                None => continue,
            };
            let mm_node = &graph.nodes[mm_idx];
            if !matches!(mm_node.op, AiOp::MatMul | AiOp::BatchMatMul) {
                continue;
            }

            // MatMul output must have exactly one consumer (this activation).
            let mm_out_tid = mm_node.outputs[0];
            let mm_consumers = consumers.get(&mm_out_tid).map_or(0, |c| c.len());
            if mm_consumers != 1 {
                continue;
            }

            let act_out_tid = act_node.outputs[0];

            let fused = AiNode::new(
                act_node.id,
                fused_op,
                mm_node.inputs.clone(),
                vec![act_out_tid],
            );

            to_remove.insert(mm_idx);
            replacements.insert(act_idx, fused);
            fused_count += 1;

            tracing::debug!(
                mm_idx,
                act_idx,
                act_out_tid,
                "MatMulActivationFusion: fused MatMul + activation"
            );
        }

        if fused_count > 0 {
            tracing::info!(fused_count, "MatMulActivationFusion: fused pairs");
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
    fn fuses_matmul_relu() {
        let mut g = empty_graph();
        g.inputs = vec![10, 11];
        g.outputs = vec![30];
        g.nodes = vec![
            AiNode::new(0, AiOp::MatMul, vec![10, 11], vec![20]),
            AiNode::new(1, AiOp::Relu, vec![20], vec![30]),
        ];

        let result = MatMulActivationFusion.run(g).expect("pass should succeed");
        assert_eq!(result.nodes.len(), 1);
        assert!(matches!(result.nodes[0].op, AiOp::MatMulRelu));
        assert_eq!(result.nodes[0].inputs, vec![10, 11]);
        assert_eq!(result.nodes[0].outputs, vec![30]);
    }

    #[test]
    fn fuses_matmul_gelu() {
        let mut g = empty_graph();
        g.inputs = vec![10, 11];
        g.outputs = vec![30];
        g.nodes = vec![
            AiNode::new(0, AiOp::MatMul, vec![10, 11], vec![20]),
            AiNode::new(1, AiOp::Gelu, vec![20], vec![30]),
        ];

        let result = MatMulActivationFusion.run(g).expect("pass should succeed");
        assert_eq!(result.nodes.len(), 1);
        assert!(matches!(result.nodes[0].op, AiOp::MatMulGelu));
    }

    #[test]
    fn fuses_matmul_silu() {
        let mut g = empty_graph();
        g.inputs = vec![10, 11];
        g.outputs = vec![30];
        g.nodes = vec![
            AiNode::new(0, AiOp::MatMul, vec![10, 11], vec![20]),
            AiNode::new(1, AiOp::Silu, vec![20], vec![30]),
        ];

        let result = MatMulActivationFusion.run(g).expect("pass should succeed");
        assert_eq!(result.nodes.len(), 1);
        assert!(matches!(result.nodes[0].op, AiOp::MatMulSilu));
    }

    #[test]
    fn skips_when_matmul_has_multiple_consumers() {
        let mut g = empty_graph();
        g.inputs = vec![10, 11];
        g.outputs = vec![30, 40];
        g.nodes = vec![
            AiNode::new(0, AiOp::MatMul, vec![10, 11], vec![20]),
            AiNode::new(1, AiOp::Relu, vec![20], vec![30]),
            AiNode::new(2, AiOp::Add, vec![20, 10], vec![40]),
        ];

        let result = MatMulActivationFusion.run(g).expect("pass should succeed");
        assert_eq!(result.nodes.len(), 3);
    }

    #[test]
    fn no_fusion_without_matmul() {
        let mut g = empty_graph();
        g.inputs = vec![10];
        g.outputs = vec![20];
        g.nodes = vec![AiNode::new(0, AiOp::Relu, vec![10], vec![20])];

        let result = MatMulActivationFusion.run(g).expect("pass should succeed");
        assert_eq!(result.nodes.len(), 1);
        assert!(matches!(result.nodes[0].op, AiOp::Relu));
    }
}
