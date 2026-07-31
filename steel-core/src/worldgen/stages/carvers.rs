use std::sync::Arc;

use crate::chunk::{
    chunk_generation_task::StaticCache2D, chunk_holder::ChunkHolder, chunk_pyramid::ChunkStep,
    status::ChunkStatus,
};
use crate::worldgen::generator::ChunkGenerator;
use crate::worldgen::generator::context::WorldGenContext;

pub(crate) fn generate(
    context: Arc<WorldGenContext>,
    _step: &ChunkStep,
    _cache: &Arc<StaticCache2D<Arc<ChunkHolder>>>,
    holder: Arc<ChunkHolder>,
) {
    let Some(chunk) = holder.try_chunk(ChunkStatus::Surface) else {
        panic!("Chunk not found at status Surface");
    };

    context.generator.apply_carvers(&chunk);
    // Generator-specific implementations normally consume their own state. This
    // central boundary also covers skipped work and future custom generators.
    chunk.clear_transient_generation_state();
}
