use divan::Bencher;
use encoderfile::{
    dev_utils::*,
    inference::{embedding, sequence_classification, token_classification},
    runtime::TokenizerService,
};
use ndarray::Array;
use rand::Rng;
use tokenizers::Encoding;

const LORUM: &str = include_str!("lorum.txt");

fn main() {
    divan::main()
}

#[divan::bench(args = [(8, 16, 384), (16, 128, 768), (64, 512, 1024)])]
fn embedding_postprocess(b: Bencher, dim: (usize, usize, usize)) {
    let tokenizer = &embedding_state().tokenizer;
    let (batch, tokens, hidden) = dim;

    // Random embeddings
    let mut rng = rand::rng();
    let data: Vec<f32> = (0..batch * tokens * hidden)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    let outputs = Array::from_shape_vec((batch, tokens, hidden), data).unwrap();

    // Dummy encodings
    let encodings = generate_dummy_encodings(tokenizer, batch, tokens);

    b.bench(|| embedding::postprocess(outputs.clone(), encodings.clone()));
}

#[divan::bench(args = [8, 16, 64])]
fn sequence_classification_postprocess(b: Bencher, batch: usize) {
    let state = sequence_classification_state();
    let config = &state.model_config;
    let n_labels = config.id2label.clone().unwrap().len();

    let mut rng = rand::rng();
    let data: Vec<f32> = (0..batch * n_labels)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();

    let outputs = Array::from_shape_vec((batch, n_labels), data).unwrap();

    b.bench(|| sequence_classification::postprocess(outputs.clone(), config));
}

#[divan::bench(args = [(8, 16), (16, 128), (64, 512)])]
fn token_classification_postprocess(b: Bencher, dim: (usize, usize)) {
    let state = token_classification_state();
    let config = &state.model_config;
    let n_labels = config.id2label.clone().unwrap().len();

    let tokenizer = &embedding_state().tokenizer;
    let (batch, tokens) = dim;

    // Random embeddings
    let mut rng = rand::rng();
    let data: Vec<f32> = (0..batch * tokens * n_labels)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    let outputs = Array::from_shape_vec((batch, tokens, n_labels), data).unwrap();

    // Dummy encodings
    let encodings = generate_dummy_encodings(tokenizer, batch, tokens);

    b.bench(|| token_classification::postprocess(outputs.clone(), encodings.clone(), config));
}

fn generate_dummy_encodings(
    tokenizer: &TokenizerService,
    batch: usize,
    max_len: usize,
) -> Vec<Encoding> {
    let inp = vec![LORUM.to_string(); batch];

    tokenizer
        .encode_text(inp)
        .unwrap()
        .into_iter()
        .map(|mut enc| {
            enc.truncate(max_len, 0, tokenizers::TruncationDirection::Right);
            enc
        })
        .collect()
}
