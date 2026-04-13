use ndarray::{Array2, Axis};
use ort::session::Session;
use parking_lot::MutexGuard;

#[macro_export]
macro_rules! prepare_inputs {
    ($encodings:ident) => {{
        let padded_token_length = $encodings[0].len();

        // Get token IDs & mask as a flattened array.
        let ids: Vec<i64> = $encodings
            .iter()
            .flat_map(|e| e.get_ids().iter().map(|i| *i as i64))
            .collect();
        let mask: Vec<i64> = $encodings
            .iter()
            .flat_map(|e| e.get_attention_mask().iter().map(|i| *i as i64))
            .collect();
        let type_ids: Vec<i64> = $encodings
            .iter()
            .flat_map(|e| e.get_type_ids().iter().map(|i| *i as i64))
            .collect();

        // Convert flattened arrays into 2-dimensional tensors of shape [N, L].
        let a_ids = ort::value::TensorRef::from_array_view((
            [$encodings.len(), padded_token_length],
            &*ids,
        ))
        .unwrap()
        .to_owned();
        let a_mask = ort::value::TensorRef::from_array_view((
            [$encodings.len(), padded_token_length],
            &*mask,
        ))
        .unwrap()
        .to_owned();
        let a_type_ids = ort::value::TensorRef::from_array_view((
            [$encodings.len(), padded_token_length],
            &*type_ids,
        ))
        .unwrap()
        .to_owned();

        (a_ids, a_mask, a_type_ids)
    }};
}

pub fn l2_normalize(mut x: Array2<f32>, axis: Axis) -> Array2<f32> {
    for mut row in x.axis_iter_mut(axis) {
        let norm = row.mapv(|v| v * v).sum().sqrt();
        if norm > 0.0 {
            row.mapv_inplace(|v| v / norm);
        }
    }
    x
}

pub fn requires_token_type_ids<'a>(session: &MutexGuard<'a, Session>) -> bool {
    session
        .inputs
        .iter()
        .any(|inp| inp.name == "token_type_ids")
}

#[macro_export]
macro_rules! run_model {
    ($session:expr, $a_ids:expr, $a_mask:expr, $a_type_ids:expr) => {{
        match $crate::inference::utils::requires_token_type_ids(&$session) {
            true => $session.run(ort::inputs!($a_ids, $a_mask, $a_type_ids)),
            false => $session.run(ort::inputs!($a_ids, $a_mask)),
        }
        .map_err(|e| {
            tracing::error!("Error running model: {:?}", e);
            $crate::error::ApiError::InternalError("Error running model")
        })
    }};
}
