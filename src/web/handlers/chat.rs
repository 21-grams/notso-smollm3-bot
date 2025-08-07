use crate::state::AppState;
use axum::{
    extract::State,
    response::Html,
};
use minijinja::{context, Environment};
use uuid::Uuid;

pub async fn index(State(state): State<AppState>) -> Html<String> {
    let session_id = Uuid::new_v4().to_string();
    
    // Initialize session
    state.sessions.write().await.create_session(&session_id);
    
    // Render template
    let mut env = Environment::new();
    env.set_loader(minijinja::path_loader("src/web/templates"));
    
    let template = env.get_template("chat.html").unwrap();
    let html = template.render(context! {
        session_id => session_id,
        thinking_mode => state.config.thinking_mode_default,
    }).unwrap();
    
    Html(html)
}

pub async fn chat_page(State(state): State<AppState>) -> Html<String> {
    index(State(state)).await
}
