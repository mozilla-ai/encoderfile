use ort::session::Session;
use parking_lot::MutexGuard;

pub type Model<'a> = MutexGuard<'a, Session>;
