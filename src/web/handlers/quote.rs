//! Quote command handler with SSE streaming

use axum::{
    response::sse::{Event, Sse},
    extract::{State, Path},
};
use futures::stream::Stream;
use std::convert::Infallible;
use crate::state::AppState;
use std::time::Duration;

/// Stream the Gospel of John 1:1-14 (Recovery Version)
pub async fn stream_quote(
    State(_state): State<AppState>,
    Path((_session_id, _message_id)): Path<(String, String)>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    tracing::info!("📖 Streaming quote for session: {}, message: {}", _session_id, _message_id);
    
    // The Gospel of John 1:1-14 (Recovery Version)
    let scripture_text = r#"# Gospel of John 1:1-14
*Recovery Version*

**1** In the beginning was the Word, and the Word was with God, and the Word was God.

**2** He was in the beginning with God.

**3** All things came into being through Him, and apart from Him not one thing came into being which has come into being.

**4** In Him was life, and the life was the light of men.

**5** And the light shines in the darkness, and the darkness did not overcome it.

**6** There came a man sent from God, whose name was John.

**7** He came for a testimony that he might testify concerning the light, that all might believe through him.

**8** He was not the light, but came that he might testify concerning the light.

**9** This was the true light which, coming into the world, enlightens every man.

**10** He was in the world, and the world came into being through Him, yet the world did not know Him.

**11** He came to His own, yet those who were His own did not receive Him.

**12** But as many as received Him, to them He gave the authority to become children of God, to those who believe into His name,

**13** Who were begotten not of blood, nor of the will of the flesh, nor of the will of man, but of God.

**14** And the Word became flesh and tabernacled among us (and we beheld His glory, glory as of the only Begotten from the Father), full of grace and reality."#;

    // Create the SSE stream with buffered output
    let stream = async_stream::stream! {
        // Split text into chunks for streaming (by sentence or verse)
        let chunks: Vec<&str> = scripture_text.split("\n\n").collect();
        let mut accumulated = String::new();
        
        for chunk in chunks {
            accumulated.push_str(chunk);
            accumulated.push_str("\n\n");
            
            // Convert markdown to HTML for proper rendering
            let html_content = markdown_to_html(&accumulated);
            
            // Send SSE message event - HTMX listens for 'message' events by default
            yield Ok(Event::default()
                .event("message")
                .data(html_content));
            
            // Delay for streaming effect
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
        
        // Send completion event
        yield Ok(Event::default()
            .event("complete")
            .data("done"));
    };
    
    Sse::new(stream)
}

/// Simple markdown to HTML converter for the quote
fn markdown_to_html(markdown: &str) -> String {
    let mut html = String::new();
    
    for line in markdown.lines() {
        if line.starts_with("# ") {
            html.push_str(&format!("<h3>{}</h3>", &line[2..]));
        } else if line.starts_with("**") && line.ends_with("**") {
            // Verse numbers
            let content = &line[2..line.len()-2];
            if let Some(rest) = line.strip_prefix("**").and_then(|s| s.strip_suffix("**")) {
                if let Some((num, text)) = rest.split_once("**") {
                    html.push_str(&format!("<p><strong>{}</strong>{}</p>", num, text));
                } else {
                    html.push_str(&format!("<p><strong>{}</strong></p>", rest));
                }
            }
        } else if line.starts_with("*") && line.ends_with("*") {
            html.push_str(&format!("<em>{}</em>", &line[1..line.len()-1]));
        } else if !line.is_empty() {
            // Parse inline bold
            let mut processed = line.to_string();
            while let Some(start) = processed.find("**") {
                if let Some(end) = processed[start+2..].find("**") {
                    let before = &processed[..start];
                    let bold = &processed[start+2..start+2+end];
                    let after = &processed[start+4+end..];
                    processed = format!("{}<strong>{}</strong>{}", before, bold, after);
                } else {
                    break;
                }
            }
            html.push_str(&format!("<p>{}</p>", processed));
        }
    }
    
    html
}
