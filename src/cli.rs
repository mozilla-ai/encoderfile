use clap_derive::{Parser, Subcommand};

#[derive(Parser)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    Serve {
        #[arg(long, default_value = "[::]")]
        grpc_hostname: String,
        #[arg(long, default_value = "50051")]
        grpc_port: String,
        #[arg(long, default_value = "0.0.0.0")]
        http_hostname: String,
        #[arg(long, default_value = "8080")]
        http_port: String,
        #[arg(long, default_value_t = false)]
        disable_grpc: bool,
        #[arg(long, default_value_t = false)]
        disable_http: bool
    },
}

#[derive(Subcommand)]
pub enum ServeCommands {
    Grpc {
        #[arg(long, default_value = "[::]")]
        hostname: String,
        #[arg(long, default_value = "50051")]
        port: String,
    },
    Http {
        #[arg(long, default_value = "0.0.0.0")]
        hostname: String,
        #[arg(long, default_value = "8080")]
        port: String,
    },
}
