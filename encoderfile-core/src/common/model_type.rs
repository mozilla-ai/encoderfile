#[derive(Debug, Clone, serde::Serialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum ModelTypeEnum {
    Embedding = 1,
    SequenceClassification = 2,
    TokenClassification = 3,
    SentenceEmbedding = 4,
}

// pub trait ModelTypeTrait {
//     const STR_REPR: &'static str;

//     fn to_string() -> &'static str;
// }

macro_rules! model_type {
    [ $( $x:ident ),* $(,)? ] => {
        $(
            pub struct $x;
        )*
    }
}
