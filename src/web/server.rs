use crate::state::AppState;
use axum::Router;
use std::net::SocketAddr;
use tower_http::services::ServeDir;
use tower_http::trace::TraceLayer;

pub async fn start_server(state: AppState) -> anyhow::Result<()> {
    let app = create_app(state.clone());
    
    let addr = SocketAddr::from(([127, 0, 0, 1], state.config.port));
    tracing::info!("🌐 Web server listening on http://{}", addr);
    
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    
    Ok(())
}

fn create_app(state: AppState) -> Router {
    Router::new()
        .merge(crate::web::routes::create_routes(state.clone()))
        .nest_service("/static", ServeDir::new("src/web/static"))
        .layer(TraceLayer::new_for_http())
        .layer(crate::web::middleware::cors_layer())
}
