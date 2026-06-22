use crate::{
    common::model_type::ModelTypeSpec,
    runtime::{Input, InputType, Task, TaskType},
};

/// Identifies the semantic role of an embedded artifact.
///
/// ## Ordering invariants
///
/// **⚠️ WARNING: The declaration order of this enum is part of the
/// encoderfile on-disk format.**
///
/// The derived `Ord` / `PartialOrd` implementation is used to establish
/// a canonical ordering of artifacts when computing byte offsets and
/// serializing the payload.
///
/// - **Do NOT reorder existing variants**
/// - **Do NOT insert new variants in the middle**
/// - **Only append new variants to the end**
///
/// Reordering existing variants will change artifact layout, break
/// determinism, and invalidate previously written encoderfiles.
///
/// If you append a new AssetKind, the new value MUST be additionally added to AssetKind::ORDERED
/// IN THE SAME ORDER as it is in the enum.
///
/// ## Evolution
///
/// New artifact kinds may be added in future format versions by
/// appending new variants. Older runtimes will safely ignore unknown
/// artifact kinds via the manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AssetKind {
    /// Model weights blob (typically large, binary).
    ModelWeights,

    /// Optional preprocessing transform source code.
    Transform,

    /// Model configuration data (e.g. architecture, hyperparameters).
    ModelConfig,

    /// Tokenizer data required for text-based models.
    Tokenizer,

    /// Optional image preprocessing configuration.
    ImagePreprocessor,
}

impl AssetKind {
    pub const ORDERED: &'static [AssetKind] = &[
        AssetKind::ModelWeights,
        AssetKind::Transform,
        AssetKind::ModelConfig,
        AssetKind::Tokenizer,
        AssetKind::ImagePreprocessor,
    ];
}

pub trait AssetPolicySpec: ModelTypeSpec + InputType + TaskType {
    fn required_assets() -> &'static [AssetKind] {
        match (Self::input_type(), Self::task_type()) {
            (Input::Text, Task::Classification) => &[
                AssetKind::ModelWeights,
                AssetKind::ModelConfig,
                AssetKind::Tokenizer,
            ],
            (Input::Text, Task::FeatureExtraction) => &[
                AssetKind::ModelWeights,
                AssetKind::ModelConfig,
                AssetKind::Tokenizer,
            ],
            (Input::Image, Task::Classification) => &[
                AssetKind::ModelWeights,
                AssetKind::ModelConfig,
                AssetKind::ImagePreprocessor,
            ],
            (Input::Image, Task::FeatureExtraction) => &[
                AssetKind::ModelWeights,
                AssetKind::ModelConfig,
                AssetKind::ImagePreprocessor,
            ],
        }
    }
    fn optional_assets() -> &'static [AssetKind] {
        match (Self::input_type(), Self::task_type()) {
            (Input::Text, Task::Classification) => &[AssetKind::Transform],
            (Input::Text, Task::FeatureExtraction) => &[AssetKind::Transform],
            (Input::Image, Task::Classification) => &[AssetKind::Transform],
            (Input::Image, Task::FeatureExtraction) => &[AssetKind::Transform],
        }
    }
}

macro_rules! asset_policy_spec {
    // Huggingface-style encoders
    (Encoder, $model_type:ident) => {
        impl AssetPolicySpec for crate::common::model_type::$model_type {}
    };
}

asset_policy_spec!(Encoder, Embedding);
asset_policy_spec!(Encoder, SequenceClassification);
asset_policy_spec!(Encoder, TokenClassification);
asset_policy_spec!(Encoder, SentenceEmbedding);
asset_policy_spec!(Encoder, ImageClassification);
