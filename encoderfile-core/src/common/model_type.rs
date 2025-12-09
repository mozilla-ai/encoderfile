macro_rules! model_type {
    [ $( $x:ident ),* $(,)? ] => {
        // create enum
        #[derive(Debug, Clone, serde::Serialize, serde::Deserialize, utoipa::ToSchema)]
        #[serde(rename_all = "snake_case")]
        pub enum ModelTypeEnum {
            $(
            $x,
            )*
        }

        $(
            pub struct $x;
        )*
    }
}

model_type![
    Embedding,
    SequenceClassification,
    TokenClassification,
    SentenceEmbedding,
];
