use ndarray::{Array2, Axis};
use ort::session::Session;
use parking_lot::MutexGuard;

pub fn softmax(x: &Array2<f32>, axis: Axis) -> Array2<f32> {
    let max_per_axis = x.map_axis(axis, |row| row.fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
    let expx = x - &max_per_axis.insert_axis(axis);
    let expx = expx.mapv(f32::exp);
    let sum = expx.sum_axis(axis).insert_axis(axis);
    &expx / &sum
}

pub fn requires_token_type_ids(session: &MutexGuard<'static, Session>) -> bool {
    session
        .inputs
        .iter()
        .any(|inp| inp.name == "token_type_ids")
}
