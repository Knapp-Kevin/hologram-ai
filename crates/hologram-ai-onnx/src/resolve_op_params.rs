//! Resolve optional ONNX op parameters from weight/input tensor shapes.
//!
//! Some ONNX attributes are optional and should be inferred from the weight
//! tensor's shape. For example, `kernel_shape` for Conv/ConvTranspose is
//! optional — when omitted, it should be inferred from the weight tensor's
//! spatial dimensions (dims[2:]).
//!
//! This pass runs immediately after ONNX import, before any optimization.
//! It ensures all ops have the parameters they need for shape propagation
//! and lowering to work correctly.

use hologram_ai_common::ir::{AiGraph, AiOp};

/// Resolve optional op parameters from tensor shapes.
///
/// Infers missing `kernel_shape` for Conv/ConvTranspose from weight tensor
/// spatial dims. Extensible to other ops with optional parameters.
pub fn resolve_op_params(graph: &mut AiGraph) {
    // Collect resolutions first (immutable borrow), then apply (mutable borrow).
    let resolutions: Vec<(usize, Resolution)> = graph
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(idx, node)| match &node.op {
            AiOp::Conv {
                kernel_shape,
                strides,
                dilations,
                ..
            } if kernel_shape.is_empty() => {
                let spatial = weight_spatial_dims(node.inputs.get(1).copied(), graph)?;
                Some((
                    idx,
                    Resolution::Conv {
                        kernel_shape: spatial.clone(),
                        strides: if strides.is_empty() {
                            vec![1; spatial.len()]
                        } else {
                            strides.clone()
                        },
                        dilations: if dilations.is_empty() {
                            vec![1; spatial.len()]
                        } else {
                            dilations.clone()
                        },
                    },
                ))
            }
            AiOp::ConvTranspose {
                kernel_shape,
                strides,
                dilations,
                ..
            } if kernel_shape.is_empty() => {
                let spatial = weight_spatial_dims(node.inputs.get(1).copied(), graph)?;
                Some((
                    idx,
                    Resolution::Conv {
                        kernel_shape: spatial.clone(),
                        strides: if strides.is_empty() {
                            vec![1; spatial.len()]
                        } else {
                            strides.clone()
                        },
                        dilations: if dilations.is_empty() {
                            vec![1; spatial.len()]
                        } else {
                            dilations.clone()
                        },
                    },
                ))
            }
            _ => None,
        })
        .collect();

    // Apply resolutions.
    for (idx, res) in resolutions {
        match res {
            Resolution::Conv {
                kernel_shape,
                strides,
                dilations,
            } => match &mut graph.nodes[idx].op {
                AiOp::Conv {
                    kernel_shape: ks,
                    strides: st,
                    dilations: dl,
                    ..
                }
                | AiOp::ConvTranspose {
                    kernel_shape: ks,
                    strides: st,
                    dilations: dl,
                    ..
                } => {
                    *ks = kernel_shape;
                    *st = strides;
                    *dl = dilations;
                }
                _ => {}
            },
        }
    }
}

enum Resolution {
    Conv {
        kernel_shape: Vec<u64>,
        strides: Vec<u64>,
        dilations: Vec<u64>,
    },
}

/// Extract spatial dimensions from a weight tensor.
///
/// For Conv weights `[C_out, C_in/groups, kH, kW, ...]`, returns `[kH, kW, ...]`.
/// For ConvTranspose weights `[C_in, C_out/groups, kH, kW, ...]`, same.
fn weight_spatial_dims(
    tid: Option<hologram_ai_common::TensorId>,
    graph: &AiGraph,
) -> Option<Vec<u64>> {
    let tid = tid?;
    let info = graph.tensor_info.get(&tid)?;
    if info.shape.len() < 3 {
        return None;
    }
    // Spatial dims start at index 2: [C_out, C_in/g, kH, kW, ...]
    let spatial: Vec<u64> = info.shape[2..]
        .iter()
        .filter_map(|d| d.as_concrete())
        .collect();
    if spatial.len() == info.shape.len() - 2 {
        Some(spatial)
    } else {
        None // Some dims weren't concrete
    }
}
