fn main() {
    let mut config = prost_build::Config::new();
    config.bytes([".onnx.TensorProto.raw_data"]);
    config
        .compile_protos(&["proto/onnx.proto"], &["proto/"])
        .expect("failed to compile onnx.proto");
}
