use mlua::prelude::*;
use ndarray::Axis;
use super::Tensor;

impl Tensor {
    #[tracing::instrument(skip_all)]
    pub fn mean_pool(&self, Tensor(mask): Tensor) -> Result<Self, LuaError> {
        assert_eq!(self.0.ndim(), mask.ndim() + 1);

        let ndim = self.0.ndim();

        // Expand mask by adding the last axis back
        let mut mask_expanded = mask.clone();
        mask_expanded = mask_expanded.insert_axis(Axis(ndim - 1));

        // Broadcast mask to full data shape
        let mask_broadcast = mask_expanded
            .broadcast(self.0.shape())
            .ok_or(LuaError::external(format!(
                "cannot broadcast shape {:?} to {:?}",
                mask_expanded.shape(),
                self.0.shape()
            )))?;

        // Multiply and sum over sequence dims (axes 1..ndim-1)
        let weighted = &self.0 * &mask_broadcast;

        // All axes except the last one and the batch axis
        let mut axes_to_reduce = Vec::new();
        for ax in 1..(ndim - 1) {
            axes_to_reduce.push(ax);
        }

        // Sum weighted values
        let mut sum = weighted.clone();
        for ax in axes_to_reduce.iter().rev() {
            sum = sum.sum_axis(Axis(*ax));
        }

        // Sum mask the same way -> counts
        let mut count = mask_expanded.clone();
        for ax in axes_to_reduce.iter().rev() {
            count = count.sum_axis(Axis(*ax));
        }

        // Final: divide elementwise
        Ok(Self(&sum / &count))
    }
}

#[test]
fn mean_pool_single_vector_no_mask() {
        // shape: (batch=1, seq=1, dim=3)
    let x = Tensor(ndarray::array![[[1.0, 2.0, 3.0]]].into_dyn());
    let mask = Tensor(ndarray::array![[1.0]].into_dyn());

    let pooled = x.mean_pool(mask).unwrap();
    assert_eq!(pooled.0, ndarray::array![[1.0, 2.0, 3.0]].into_dyn());
}

#[test]
fn mean_pool_two_tokens_equal_weight() {
        // shape: (1, 2, 3)
    let x = Tensor(ndarray::array![[[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]].into_dyn());

    let mask = Tensor(ndarray::array![[1.0, 1.0]].into_dyn());

    let pooled = x.mean_pool(mask).unwrap();
    let expected = ndarray::array![[2.0, 2.0, 2.0]].into_dyn();

    assert_allclose(&pooled.0, &expected);
}

#[test]
fn mean_pool_ignores_masked_tokens() {
        // shape: (1, 3, 2)
    // Only the first and last token should count.
    let x = Tensor(
        ndarray::array![[
            [10.0, 0.0],
            [99.0, 99.0], // masked out
            [20.0, 0.0]
        ]]
        .into_dyn(),
    );

    let mask = Tensor(ndarray::array![[1.0, 0.0, 1.0]].into_dyn());

    let pooled = x.mean_pool(mask).unwrap();
    let expected = ndarray::array![[(10.0 + 20.0) / 2.0, 0.0]].into_dyn();

    assert_allclose(&pooled.0, &expected);
}

#[test]
fn mean_pool_batch_mode() {
        // shape: (2, 2, 2)
    let x = Tensor(
        ndarray::array![
            [[1.0, 1.0], [3.0, 3.0]], // batch 0
            [[2.0, 4.0], [4.0, 2.0]], // batch 1
        ]
        .into_dyn(),
    );

    let mask = Tensor(ndarray::array![[1.0, 1.0], [1.0, 0.0],].into_dyn());

    let pooled = x.mean_pool(mask).unwrap();

    let expected = ndarray::array![[(1.0 + 3.0) / 2.0, (1.0 + 3.0) / 2.0], [2.0, 4.0]].into_dyn();

    assert_allclose(&pooled.0, &expected);
}

#[test]
fn mean_pool_mask_broadcasting() {
        let x = Tensor(
        ndarray::array![[
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]]
        ]]
        .into_dyn(),
    );

    let mask = Tensor(ndarray::array![[[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]].into_dyn());

    let pooled = x.mean_pool(mask).unwrap();

    // Compute manually:
    // First inner seq: avg of [1,2] and [4,5]
    // Second inner seq isn't separate — everything is reduced together.
    //
    // Values included:
    //   1.0, 2.0, 4.0, 5.0   (mask=1)
    // and the same duplicated for the second feature.
    let expected = ndarray::array![[3.0, 3.0]].into_dyn(); // (1,2)

    assert_allclose(&pooled.0, &expected);
}

pub fn assert_allclose(a: &ndarray::ArrayD<f32>, b: &ndarray::ArrayD<f32>) {
    let tol = 1e-6;
    assert_eq!(
        a.shape(),
        b.shape(),
        "shape mismatch: {:?} vs {:?}",
        a.shape(),
        b.shape()
    );
    let a_slice = a.as_slice().unwrap();
    let b_slice = b.as_slice().unwrap();
    for (i, (x, y)) in a_slice.iter().zip(b_slice.iter()).enumerate() {
        let diff = (x - y).abs();
        assert!(
            diff <= tol,
            "mismatch at index {i}: {x} vs {y} (diff {diff})"
        );
    }
}
