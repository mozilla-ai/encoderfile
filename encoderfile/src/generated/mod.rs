pub mod embedding{
    tonic::include_proto!("encoderfile.embedding");
}

pub mod sequence_classification {
    tonic::include_proto!("encoderfile.sequence_classification");
}

pub mod token_classification {
    tonic::include_proto!("encoderfile.token_classification");
}

pub mod sentence_embedding {
    tonic::include_proto!("encoderfile.sentence_embedding");
}

pub mod token {
    tonic::include_proto!("encoderfile.token");
}

pub mod metadata {
    tonic::include_proto!("encoderfile.metadata");
}
