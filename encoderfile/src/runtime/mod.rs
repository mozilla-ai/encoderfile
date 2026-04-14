use ort::session::Session;
use parking_lot::MutexGuard;

mod loader;
mod session;
mod state;
mod tokenizer;

pub use loader::{EncoderfileLoader, load_assets};
pub use session::{ORTExecutionProvider, ORTSessionBuilder};
pub use state::{AppState, EncoderfileState};
pub use tokenizer::TokenizerService;

pub type Model<'a> = MutexGuard<'a, Session>;
