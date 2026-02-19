use encoderfile::transforms::{DEFAULT_LIBS, Postprocessor};
use ndarray::{Array2, Array3};
use rand::Rng;

fn main() {
    divan::main()
}

fn get_random_2d(x: usize, y: usize) -> Array2<f32> {
    let mut rng = rand::rng();
    let vals: Vec<f32> = (0..(x * y)).map(|_| rng.random()).collect();

    Array2::from_shape_vec((x, y), vals).expect("Failed to create shape vec")
}

fn get_random_3d(x: usize, y: usize, z: usize) -> Array3<f32> {
    let mut rng = rand::rng();
    let vals: Vec<f32> = (0..(x * y * z)).map(|_| rng.random()).collect();

    Array3::from_shape_vec((x, y, z), vals).expect("Failed to create shape vec")
}

#[divan::bench(args = [(16, 16, 16), (32, 128, 384), (32, 256, 768)])]
fn bench_embedding_l2_normalization(bencher: divan::Bencher, (x, y, z): (usize, usize, usize)) {
    let engine = encoderfile::transforms::EmbeddingTransform::new(
        DEFAULT_LIBS.to_vec(),
        Some(include_str!("../../transforms/embedding/l2_normalize_embeddings.lua").to_string()),
    )
    .unwrap();

    let test_tensor = get_random_3d(x, y, z);

    bencher.bench_local(|| {
        engine.postprocess(test_tensor.clone()).unwrap();
    });
}

#[divan::bench(args = [(16, 2), (32, 8), (128, 32)])]
fn bench_seq_cls_softmax(bencher: divan::Bencher, (x, y): (usize, usize)) {
    let engine = encoderfile::transforms::SequenceClassificationTransform::new(
        DEFAULT_LIBS.to_vec(),
        Some(
            include_str!("../../transforms/sequence_classification/softmax_logits.lua").to_string(),
        ),
    )
    .unwrap();

    let test_tensor = get_random_2d(x, y);

    bencher.bench_local(|| {
        engine.postprocess(test_tensor.clone()).unwrap();
    });
}

#[divan::bench(args = [(16, 16, 2), (32, 128, 8), (128, 256, 32)])]
fn bench_tok_cls_softmax(bencher: divan::Bencher, (x, y, z): (usize, usize, usize)) {
    let engine = encoderfile::transforms::TokenClassificationTransform::new(
        DEFAULT_LIBS.to_vec(),
        Some(include_str!("../../transforms/token_classification/softmax_logits.lua").to_string()),
    )
    .unwrap();

    let test_tensor = get_random_3d(x, y, z);

    bencher.bench_local(|| {
        engine.postprocess(test_tensor.clone()).unwrap();
    });
}
