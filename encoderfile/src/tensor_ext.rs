use ndarray::{ArrayBase, Data, Dimension};

pub(crate) trait TensorExt {
    fn argmax_scalar(&self) -> Option<usize>;
}

impl<S, D> TensorExt for ArrayBase<S, D>
where
    S: Data,
    S::Elem: PartialOrd,
    D: Dimension,
{
    fn argmax_scalar(&self) -> Option<usize> {
        self.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
    }
}
