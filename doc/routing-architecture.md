# Routing Architecture

## Overview
The web server uses Axum framework for HTTP routing with a clear separation of concerns between static file serving and application routes.

## Route Organization

### Static Files (`/static/*`)
- **Location**: Registered in `src/web/server.rs`
- **Handler**: `tower_http::services::ServeDir`
- **Purpose**: Serves CSS, JavaScript, images, and other static assets
- **Path**: Files served from `src/web/static/` directory

### Application Routes
Defined in `src/web/routes.rs` and organized into:

#### Pages
- `/` - Main index page (chat interface)
- `/chat` - Chat page

#### API Endpoints
- `POST /api/chat` - Send message to the chatbot
- `GET /api/stream/{session_id}` - Server-sent events for streaming responses
- `POST /api/toggle-thinking` - Toggle thinking/reasoning display

#### Health Check
- `GET /health` - Application health status

## Middleware Stack
Applied in `server.rs` in the following order:
1. Application routes (merged)
2. Static file serving
3. HTTP tracing layer
4. CORS layer

## Common Issues and Solutions

### Route Conflict Resolution
**Issue**: "Invalid route... Insertion failed due to conflict with previously registered route"

**Cause**: Attempting to register the same route pattern multiple times, typically when:
- Same route defined in multiple places
- Static file route registered both in `routes.rs` and `server.rs`

**Solution**: 
- Keep static file serving exclusively in `server.rs`
- Keep application routes in `routes.rs`
- Avoid duplicate route registrations

## Best Practices
1. **Single Responsibility**: Each module handles specific route types
2. **Clear Hierarchy**: Static files served at app level, not mixed with application routes
3. **Middleware Order**: Apply middleware from most specific to most general
4. **Path Consistency**: Use consistent path patterns across the application
