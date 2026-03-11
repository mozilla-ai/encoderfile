macro_rules! model_type {
    [ $( $x:ident ),* $(,)? ] => {
        // create enum
        #[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize, utoipa::ToSchema, schemars::JsonSchema)]
        #[serde(rename_all = "snake_case")]
        pub enum ModelType {
            $(
            $x,
            )*
        }

        impl std::str::FromStr for ModelType {
            type Err = String;

            fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
                serde_json::from_value::<ModelType>(serde_json::Value::String(s.to_string()))
                    .map_err(|_| format!("Invalid model type: {}", s))
            }
        }

        impl std::fmt::Display for ModelType {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                serde_json::to_value(self)
                    .map_err(|_| std::fmt::Error)?
                    .as_str()
                    .ok_or(std::fmt::Error)?
                    .fmt(f)
            }
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

pub trait ModelTypeSpec: Send + Sync + Clone + std::fmt::Debug + 'static {
    fn enum_val() -> ModelType;
}
