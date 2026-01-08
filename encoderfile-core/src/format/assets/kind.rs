#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum AssetKind {
    ModelWeights,
    Transform,
    ModelConfig,
    Tokenizer,
}
