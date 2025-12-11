use encoderfile_core::transforms::{
    EmbeddingTransform, Postprocessor, SequenceClassificationTransform,
    TokenClassificationTransform,
};
use ndarray::{Array2, Array3, Axis};
use ort::tensor::ArrayExtensions;

#[test]
fn test_l2_normalization() {
    let engine = EmbeddingTransform::new(Some(
        include_str!("../../../transforms/embedding/l2_normalize_embeddings.lua").to_string(),
    ))
    .expect("Failed to create engine");

    let test_arr = Array3::<f32>::from_elem((8, 16, 36), 1.0);

    let norms = test_arr
        .map_axis(Axis(2), |v| (v.mapv(|x| x.powi(2)).sum()).sqrt())
        .insert_axis(Axis(2)); // shape: [batch_size, n_tokens, 1]

    // Avoid divide-by-zero by adding small epsilon
    let eps = 1e-12;
    let gold = &test_arr / &(&norms + eps);

    let test = engine.postprocess(test_arr).expect("Didn't read");

    assert_eq!(gold, test)
}

#[test]
fn test_softmax_sequence_cls() {
    let engine = SequenceClassificationTransform::new(Some(
        include_str!("../../../transforms/sequence_classification/softmax_logits.lua").to_string(),
    ))
    .expect("Failed to create engine");

    // run on array of shape [batch_size, n_labels]
    let test_arr = Array2::<f32>::from_elem((8, 16), 1.0);

    let softmax_gold = test_arr.softmax(Axis(1));

    let softmax_test = engine
        .postprocess(test_arr)
        .expect("Failed to compute softmax");

    assert_eq!(softmax_gold, softmax_test);
}

#[test]
fn test_softmax_token_cls() {
    let engine = TokenClassificationTransform::new(Some(
        include_str!("../../../transforms/token_classification/softmax_logits.lua").to_string(),
    ))
    .expect("Failed to create engine");

    // run on array of shape [batch_size, n_tokens, n_labels]
    let test_arr = Array3::<f32>::from_elem((8, 16, 2), 1.0);

    let softmax_gold = test_arr.softmax(Axis(2));

    let softmax_test = engine
        .postprocess(test_arr)
        .expect("Failed to compute softmax");

    assert_eq!(softmax_gold, softmax_test);
}
