macro_rules! model_type {
    [ $( $x:ident ),* $(,)? ] => {
        // create enum
        #[derive(Debug, Clone, serde::Serialize, serde::Deserialize, utoipa::ToSchema)]
        #[serde(rename_all = "snake_case")]
        pub enum ModelType {
            $(
            $x,
            )*
        }

        $(
            #[derive(Debug, Clone)]
            pub struct $x;

            impl ModelTypeSpec for $x {
                fn enum_val() -> ModelType {
                    ModelType::$x
                }
            }
        )*
    }
}

model_type![
    Embedding,
    SequenceClassification,
    TokenClassification,
    SentenceEmbedding,
];

pub trait ModelTypeSpec: Send + Sync + std::fmt::Debug {
    fn enum_val() -> ModelType;
}
