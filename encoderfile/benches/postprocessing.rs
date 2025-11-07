use encoderfile::{inference::embedding, runtime::tokenizer::{encode_text, get_tokenizer}};
use ndarray::Array3;
use tokenizers::Encoding;
use divan::Bencher;

const LORUM: &str = include_str!("lorum.txt");

fn main() {
    divan::main()
}

#[divan::bench(args = [(8, 16, 384), (16, 128, 768), (64, 512, 1024)])]
fn embedding_postprocess(b: Bencher, dim: (usize, usize, usize)) {
    let (batch, tokens, hidden) = dim;
    let (outputs, encodings) = sample_inputs(batch, tokens, hidden);
    b.bench(|| embedding::postprocess(outputs.clone(), encodings.clone(), false));
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
    let encodings = generate_dummy_encodings(batch, tokens);

    (outputs, encodings)
}

fn generate_dummy_encodings(batch: usize, max_len: usize) -> Vec<Encoding> {
    let tokenizer = get_tokenizer();

    let inp = vec![LORUM.to_string(); batch];

    encode_text(&tokenizer, inp)
        .unwrap()
        .into_iter()
        .map(|mut enc| {
            enc.truncate(max_len, 0, tokenizers::TruncationDirection::Right);
            enc
        })
        .collect()
}
