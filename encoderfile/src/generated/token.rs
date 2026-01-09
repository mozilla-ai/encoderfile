use crate::common;

tonic::include_proto!("encoderfile.token");

impl From<common::TokenInfo> for TokenInfo {
    fn from(val: common::TokenInfo) -> Self {
        crate::generated::token::TokenInfo {
            token: val.token,
            token_id: val.token_id,
            start: (val.start as u32),
            end: (val.end as u32),
        }
    }
}
