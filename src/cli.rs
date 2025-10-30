use clap_derive::{Parser, Subcommand};

#[derive(Parser)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands
}

#[derive(Subcommand)]
pub enum Commands {
    Serve {
        #[command(subcommand)]
        command: ServeCommands
    }
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
