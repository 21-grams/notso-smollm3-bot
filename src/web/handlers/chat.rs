use crate::state::AppState;
// use crate::services::template::TemplateEngine; // Currently unused
use axum::{
    extract::State,
    response::Html,
};
use uuid::Uuid;
// use minijinja::context; // Currently unused

pub async fn index(State(state): State<AppState>) -> Html<String> {
    let session_id = Uuid::now_v7().to_string();
    
    // Initialize session
    state.sessions.write().await.create_session(&session_id);
    
    // Read and render the single chat template
    let html = if let Ok(tmpl_content) = std::fs::read_to_string("src/web/templates/chat.html") {
        // Replace template variables
        tmpl_content
            .replace("{{ session_id }}", &session_id)
            .replace("{% if thinking_mode %}checked{% endif %}", 
                     if state.config.thinking_mode_default { "checked" } else { "" })
    } else {
        // Fallback error page
        format!(
            r#"<!DOCTYPE html>
            <html>
            <head><title>Error</title></head>
            <body>
                <h1>Error loading chat template</h1>
                <p>Please ensure chat.html exists in the templates directory.</p>
            </body>
            </html>"#
        )
    };
    
    Html(html)
}

pub async fn chat_page(State(state): State<AppState>) -> Html<String> {
    index(State(state)).await
}
