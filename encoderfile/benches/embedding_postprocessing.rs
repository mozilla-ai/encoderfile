use encoderfile::inference::embedding::postprocess;
use ndarray::Array3;
use tokenizers::Encoding;
use divan::Bencher;

fn main() {
    divan::main()
}

#[divan::bench(args = [(8, 16, 384), (16, 128, 768), (64, 512, 1024)])]
fn embedding_postprocess(b: Bencher, dim: (usize, usize, usize)) {
    let (batch, tokens, hidden) = dim;
    let (outputs, encodings) = sample_inputs(batch, tokens, hidden);
    b.bench(|| postprocess(outputs.clone(), encodings.clone(), false));
}

fn sample_inputs(batch: usize, tokens: usize, hidden: usize) -> (Array3<f32>, Vec<Encoding>) {
    use ndarray::Array;
    use rand::Rng;

    // Random embeddings
    let mut rng = rand::rng();
    let data: Vec<f32> = (0..batch * tokens * hidden)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    let outputs = Array::from_shape_vec((batch, tokens, hidden), data).unwrap();

    // Dummy encodings
    let encodings = (0..batch)
        .map(|_| {
            Encoding::default()
        })
        .collect();

    (outputs, encodings)
}
