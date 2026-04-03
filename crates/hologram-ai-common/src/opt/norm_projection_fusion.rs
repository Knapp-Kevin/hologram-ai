//! Norm + Projection fusion (Plan 054).
//!
//! Detects `[Add +] RmsNorm → 2+ MatMul` (multi-way projection) and fuses
//! into `FusedNormProjection`.
//!
//! # Patterns
//!
//! ## Pattern A: Norm → multi-way projection
//! ```text
//! normed = RmsNorm(x, weight, eps)
//!   → MatMul(normed, W_a) → a
//!   → MatMul(normed, W_b) → b
//!  [→ MatMul(normed, W_c) → c]
//! ```
//!
//! ## Pattern B: Add + Norm → multi-way projection
//! ```text
//! normed = FusedLayerNormResidual(x, residual, weight, eps)
//!   → MatMul(normed, W_a) → a
//!   → MatMul(normed, W_b) → b
//!  [→ MatMul(normed, W_c) → c]
//! ```
//!
//! Fused into:
//! ```text
//! concat_out = FusedNormProjection(x, [residual,] weight, W_concat)
//!   → Slice(0..n_a) → a
//!   → Slice(n_a..n_a+n_b) → b
//!  [→ Slice(n_a+n_b..n_total) → c]
//! ```
//!
//! The fused kernel normalizes in a stack buffer (no arena allocation),
//! then projects via a single GEMV into the concatenated output.
//! Downstream Slice nodes split the output (zero-copy at M=1).
//!
//! Constraint: the norm output must have no consumers besides the projections.
//! Only f32 weights are concatenated (Q4 weights skipped to avoid rkyv overflow).

use super::pipeline::Pass;
use crate::ir::{AiGraph, AiNode, AiOp, AiParam, Dim, DType, SemanticHint, TensorId, TensorInfo};
use std::collections::{HashMap, HashSet};

pub struct NormProjectionFusion;

impl Pass for NormProjectionFusion {
    fn name(&self) -> &str {
        "NormProjectionFusion"
    }

    fn run(&self, mut graph: AiGraph) -> anyhow::Result<AiGraph> {
        // Map: tensor_id → list of (consuming_node_idx, input_position).
        let mut consumers: HashMap<TensorId, Vec<(usize, usize)>> = HashMap::new();
        for (i, n) in graph.nodes.iter().enumerate() {
            for (pos, &tid) in n.inputs.iter().enumerate() {
                consumers.entry(tid).or_default().push((i, pos));
            }
        }

        let mut to_remove: HashSet<usize> = HashSet::new();
        let mut new_node_groups: Vec<(usize, Vec<AiNode>)> = Vec::new();
        let mut fused_count: u32 = 0;

        let mut next_node_id = graph.nodes.iter().map(|n| n.id).max().unwrap_or(0) + 1;
        let mut next_tid = graph
            .nodes
            .iter()
            .flat_map(|n| n.outputs.iter().chain(n.inputs.iter()))
            .copied()
            .max()
            .unwrap_or(0)
            + 1;
        if let Some(max_param_tid) = graph.params.keys().max() {
            next_tid = next_tid.max(*max_param_tid + 1);
        }

        for (norm_idx, norm_node) in graph.nodes.iter().enumerate() {
            let (epsilon, has_residual_add) = match &norm_node.op {
                AiOp::RmsNorm { epsilon } => (*epsilon as f64, false),
                AiOp::FusedLayerNormResidual { epsilon } => (*epsilon as f64, true),
                _ => continue,
            };

            if to_remove.contains(&norm_idx) {
                continue;
            }

            let norm_out = match norm_node.outputs.first() {
                Some(&tid) => tid,
                None => continue,
            };

            let norm_consumers = match consumers.get(&norm_out) {
                Some(c) => c,
                None => continue,
            };

            // Collect MatMul consumers where norm output is input[0].
            let matmul_consumers: Vec<(usize, TensorId, TensorId)> = norm_consumers
                .iter()
                .filter_map(|&(node_idx, input_pos)| {
                    if input_pos != 0 {
                        return None;
                    }
                    let node = &graph.nodes[node_idx];
                    if !matches!(node.op, AiOp::MatMul) || node.inputs.len() < 2 {
                        return None;
                    }
                    if to_remove.contains(&node_idx) {
                        return None;
                    }
                    let weight_tid = node.inputs[1];
                    let out_tid = *node.outputs.first()?;
                    Some((node_idx, weight_tid, out_tid))
                })
                .collect();

            if matmul_consumers.len() < 2 {
                continue;
            }

            // Norm output must ONLY be consumed by these MatMuls.
            if norm_consumers.len() != matmul_consumers.len() {
                continue;
            }

            // All weights must be f32 parameters with concrete 2D shapes.
            let mut weight_infos: Vec<(TensorId, usize, usize)> = Vec::new(); // (tid, k, n)
            let mut all_valid = true;
            for &(_, weight_tid, _) in &matmul_consumers {
                if !graph.params.contains_key(&weight_tid) {
                    all_valid = false;
                    break;
                }
                match get_2d_shape(&graph, weight_tid) {
                    Some((k, n)) if k >= 256 && n >= 64 => {
                        weight_infos.push((weight_tid, k, n));
                    }
                    _ => {
                        all_valid = false;
                        break;
                    }
                }
            }
            if !all_valid {
                continue;
            }

            // All K dims must match.
            let k = weight_infos[0].1;
            if !weight_infos.iter().all(|w| w.1 == k) {
                continue;
            }

            // Only fuse f32 weights (skip Q4/Q8 to avoid rkyv overflow).
            let all_f32 = matmul_consumers.iter().all(|&(_, weight_tid, _)| {
                graph
                    .tensor_info
                    .get(&weight_tid)
                    .is_some_and(|info| info.storage_dtype == DType::F32)
            });
            if !all_f32 {
                tracing::trace!(
                    norm_idx,
                    "NormProjectionFusion: skipping — non-f32 weights (Q4/Q8)"
                );
                continue;
            }

            let split_sizes: Vec<usize> = weight_infos.iter().map(|w| w.2).collect();
            let n_total: usize = split_sizes.iter().sum();

            // Concatenate weights along N axis.
            let concat_result = concat_weights_along_n(&graph, &matmul_consumers, k);
            let concat_bytes = match concat_result {
                Ok(bytes) => bytes,
                Err(e) => {
                    tracing::warn!(
                        norm_idx,
                        "NormProjectionFusion: weight concat failed: {e}"
                    );
                    continue;
                }
            };

            // Allocate TIDs for concat weight and fused output.
            let concat_weight_tid = next_tid;
            next_tid += 1;
            let fused_output_tid = next_tid;
            next_tid += 1;

            // Register concat weight.
            let weight_shape = crate::ir::shape_from_concrete(&[k as u64, n_total as u64]);
            let weight_info = TensorInfo {
                shape: weight_shape,
                logical_dtype: DType::F32,
                storage_dtype: DType::F32,
                quant: hologram_ai_quant::QuantDescriptor::none(),
                known_i64_values: None,
                semantic: SemanticHint::Unknown,
            };
            graph.params.insert(
                concat_weight_tid,
                AiParam::Inline {
                    data: concat_bytes,
                    info: weight_info.clone(),
                },
            );
            graph.tensor_info.insert(concat_weight_tid, weight_info);

            // Register fused output tensor info.
            let fused_output_info = {
                let input_tid = norm_node.inputs[0];
                let mut shape = graph
                    .tensor_info
                    .get(&input_tid)
                    .map(|i| i.shape.clone())
                    .unwrap_or_else(|| crate::ir::shape_from_concrete(&[1]));
                if let Some(last) = shape.last_mut() {
                    *last = Dim::Concrete(n_total as u64);
                }
                TensorInfo {
                    shape,
                    logical_dtype: DType::F32,
                    storage_dtype: DType::F32,
                    quant: hologram_ai_quant::QuantDescriptor::none(),
                    known_i64_values: None,
                    semantic: SemanticHint::Unknown,
                }
            };
            graph.tensor_info.insert(fused_output_tid, fused_output_info);

            // Build fused node inputs.
            let fused_inputs = if has_residual_add {
                // FusedLayerNormResidual inputs: [x, residual, weight]
                vec![
                    norm_node.inputs[0], // x
                    norm_node.inputs[1], // residual
                    norm_node.inputs[2], // norm_weight
                    concat_weight_tid,   // proj_weight
                ]
            } else {
                // RmsNorm inputs: [x, weight]
                vec![
                    norm_node.inputs[0], // x
                    norm_node.inputs[1], // norm_weight
                    concat_weight_tid,   // proj_weight
                ]
            };

            let fused_node_id = next_node_id;
            next_node_id += 1;
            let fused_node = AiNode::new(
                fused_node_id,
                AiOp::FusedNormProjection {
                    epsilon,
                    split_sizes: split_sizes.clone(),
                    has_residual_add,
                },
                fused_inputs,
                vec![fused_output_tid],
            );

            let mut new_nodes = vec![fused_node];

            // Create Slice nodes for each original output.
            let mut col_offset = 0i64;
            for (i, &(_, _, orig_out_tid)) in matmul_consumers.iter().enumerate() {
                let slice_node_id = next_node_id;
                next_node_id += 1;
                let start = col_offset;
                let end = col_offset + split_sizes[i] as i64;

                let slice_node = AiNode::new(
                    slice_node_id,
                    AiOp::Slice {
                        axes: vec![-1],
                        starts: vec![start],
                        ends: vec![end],
                        steps: vec![1],
                    },
                    vec![fused_output_tid],
                    vec![orig_out_tid],
                );

                // Preserve original output tensor info.
                new_nodes.push(slice_node);
                col_offset = end;
            }

            // Mark norm + matmul nodes for removal.
            to_remove.insert(norm_idx);
            for &(mm_idx, _, _) in &matmul_consumers {
                to_remove.insert(mm_idx);
            }
            let insert_idx = norm_idx;
            new_node_groups.push((insert_idx, new_nodes));
            fused_count += 1;

            tracing::debug!(
                norm_idx,
                n_projections = matmul_consumers.len(),
                ?split_sizes,
                n_total,
                has_residual_add,
                "NormProjectionFusion: fused Norm + {}-way projection",
                matmul_consumers.len(),
            );
        }

        if fused_count > 0 {
            tracing::info!(
                fused_count,
                "NormProjectionFusion: fused Norm+Projection groups"
            );
        }

        if !to_remove.is_empty() {
            new_node_groups.sort_by_key(|(idx, _)| *idx);
            let mut result_nodes = Vec::with_capacity(graph.nodes.len());
            let mut insert_iter = new_node_groups.into_iter().peekable();

            for (i, node) in graph.nodes.into_iter().enumerate() {
                while let Some((insert_idx, _)) = insert_iter.peek() {
                    if *insert_idx <= i {
                        let (_, nodes) = insert_iter.next().expect("peeked");
                        result_nodes.extend(nodes);
                    } else {
                        break;
                    }
                }
                if to_remove.contains(&i) {
                    continue;
                }
                result_nodes.push(node);
            }
            for (_, nodes) in insert_iter {
                result_nodes.extend(nodes);
            }

            graph.nodes = result_nodes;
            graph.invalidate_topo_cache();
        }

        Ok(graph)
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn get_2d_shape(graph: &AiGraph, tid: TensorId) -> Option<(usize, usize)> {
    let info = graph.tensor_info.get(&tid)?;
    let dims: Vec<usize> = info
        .shape
        .iter()
        .filter_map(|d| match d {
            Dim::Concrete(n) => Some(*n as usize),
            _ => None,
        })
        .collect();
    if dims.len() == 2 {
        Some((dims[0], dims[1]))
    } else {
        None
    }
}

fn param_bytes(param: &AiParam) -> anyhow::Result<Vec<u8>> {
    use anyhow::Context;
    match param {
        AiParam::Inline { data, .. } => Ok(data.clone()),
        AiParam::Mmap {
            path, offset, len, ..
        } => {
            use std::io::{Read, Seek, SeekFrom};
            let mut f = std::fs::File::open(path)
                .with_context(|| format!("opening mmap param at {path:?}"))?;
            f.seek(SeekFrom::Start(*offset))?;
            let mut buf = vec![0u8; *len as usize];
            f.read_exact(&mut buf)?;
            Ok(buf)
        }
    }
}

/// Concatenate weight matrices along the N (column) axis.
fn concat_weights_along_n(
    graph: &AiGraph,
    matmul_consumers: &[(usize, TensorId, TensorId)],
    k: usize,
) -> anyhow::Result<Vec<u8>> {
    let ns: Vec<usize> = matmul_consumers
        .iter()
        .map(|&(_, weight_tid, _)| {
            graph
                .tensor_info
                .get(&weight_tid)
                .and_then(|i| i.shape.last())
                .and_then(|d| d.as_concrete())
                .map(|c| c as usize)
                .unwrap_or(0)
        })
        .collect();
    let n_total: usize = ns.iter().sum();

    let weight_data: Vec<Vec<u8>> = matmul_consumers
        .iter()
        .map(|&(_, weight_tid, _)| {
            let param = graph
                .params
                .get(&weight_tid)
                .ok_or_else(|| anyhow::anyhow!("weight param not found for tid {}", weight_tid))?;
            param_bytes(param)
        })
        .collect::<anyhow::Result<_>>()?;

    let mut result = vec![0u8; k * n_total * 4];
    for row in 0..k {
        let mut col_offset = 0usize;
        for (m_idx, &n_i) in ns.iter().enumerate() {
            let src_start = row * n_i * 4;
            let src_end = src_start + n_i * 4;
            let dst_start = (row * n_total + col_offset) * 4;
            if src_end <= weight_data[m_idx].len() {
                result[dst_start..dst_start + n_i * 4]
                    .copy_from_slice(&weight_data[m_idx][src_start..src_end]);
            }
            col_offset += n_i;
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{AiGraph, AiNode, AiOp, AiParam, TensorInfo};

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

    fn f32_weight(k: usize, n: usize) -> (AiParam, TensorInfo) {
        let data = vec![0u8; k * n * 4];
        let info = TensorInfo {
            shape: crate::ir::shape_from_concrete(&[k as u64, n as u64]),
            logical_dtype: DType::F32,
            storage_dtype: DType::F32,
            quant: hologram_ai_quant::QuantDescriptor::none(),
            known_i64_values: None,
            semantic: SemanticHint::Unknown,
        };
        (AiParam::Inline { data, info: info.clone() }, info)
    }

    #[test]
    fn fuses_rmsnorm_with_two_matmul_consumers() {
        let mut g = empty_graph();
        // x=10, norm_weight=11, W_a=12, W_b=13
        // RmsNorm(x, norm_weight) → normed=20
        // MatMul(normed, W_a) → a=30
        // MatMul(normed, W_b) → b=31
        g.inputs = vec![10];
        g.outputs = vec![30, 31];

        let (param_a, info_a) = f32_weight(512, 256);
        let (param_b, info_b) = f32_weight(512, 256);
        g.params.insert(12, param_a);
        g.params.insert(13, param_b);
        g.tensor_info.insert(12, info_a);
        g.tensor_info.insert(13, info_b);

        // Input x shape: [1, 512]
        g.tensor_info.insert(10, TensorInfo {
            shape: crate::ir::shape_from_concrete(&[1, 512]),
            logical_dtype: DType::F32,
            storage_dtype: DType::F32,
            quant: hologram_ai_quant::QuantDescriptor::none(),
            known_i64_values: None,
            semantic: SemanticHint::Unknown,
        });

        g.nodes = vec![
            AiNode::new(0, AiOp::RmsNorm { epsilon: 1e-5 }, vec![10, 11], vec![20]),
            AiNode::new(1, AiOp::MatMul, vec![20, 12], vec![30]),
            AiNode::new(2, AiOp::MatMul, vec![20, 13], vec![31]),
        ];

        let result = NormProjectionFusion.run(g).expect("pass should succeed");
        // Should produce: 1 FusedNormProjection + 2 Slices = 3 nodes
        assert_eq!(result.nodes.len(), 3, "should have fused_node + 2 slices");
        assert!(
            matches!(result.nodes[0].op, AiOp::FusedNormProjection { .. }),
            "first node should be FusedNormProjection, got {:?}",
            result.nodes[0].op
        );
        assert!(
            matches!(result.nodes[1].op, AiOp::Slice { .. }),
            "second node should be Slice"
        );
        assert!(
            matches!(result.nodes[2].op, AiOp::Slice { .. }),
            "third node should be Slice"
        );

        // Check FusedNormProjection params.
        if let AiOp::FusedNormProjection {
            epsilon,
            split_sizes,
            has_residual_add,
        } = &result.nodes[0].op
        {
            assert!((*epsilon - 1e-5_f64).abs() < 1e-10);
            assert_eq!(split_sizes, &[256, 256]);
            assert!(!has_residual_add);
        }

        // Slices should output the original tids.
        assert_eq!(result.nodes[1].outputs, vec![30]);
        assert_eq!(result.nodes[2].outputs, vec![31]);
    }

    #[test]
    fn skips_single_matmul_consumer() {
        let mut g = empty_graph();
        g.inputs = vec![10];
        g.outputs = vec![30];

        let (param_a, info_a) = f32_weight(512, 256);
        g.params.insert(12, param_a);
        g.tensor_info.insert(12, info_a);

        g.nodes = vec![
            AiNode::new(0, AiOp::RmsNorm { epsilon: 1e-5 }, vec![10, 11], vec![20]),
            AiNode::new(1, AiOp::MatMul, vec![20, 12], vec![30]),
        ];

        let result = NormProjectionFusion.run(g).expect("pass should succeed");
        assert_eq!(result.nodes.len(), 2, "should not fuse single consumer");
    }

    #[test]
    fn skips_when_norm_has_non_matmul_consumer() {
        let mut g = empty_graph();
        g.inputs = vec![10];
        g.outputs = vec![30, 31, 40];

        let (param_a, info_a) = f32_weight(512, 256);
        let (param_b, info_b) = f32_weight(512, 256);
        g.params.insert(12, param_a);
        g.params.insert(13, param_b);
        g.tensor_info.insert(12, info_a);
        g.tensor_info.insert(13, info_b);

        g.nodes = vec![
            AiNode::new(0, AiOp::RmsNorm { epsilon: 1e-5 }, vec![10, 11], vec![20]),
            AiNode::new(1, AiOp::MatMul, vec![20, 12], vec![30]),
            AiNode::new(2, AiOp::MatMul, vec![20, 13], vec![31]),
            AiNode::new(3, AiOp::Add, vec![20, 10], vec![40]),
        ];

        let result = NormProjectionFusion.run(g).expect("pass should succeed");
        assert_eq!(
            result.nodes.len(),
            4,
            "should not fuse with non-MatMul consumer"
        );
    }
}
