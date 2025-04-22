from fastapi.openapi.utils import get_openapi

def custom_openapi(app):
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="AI RAG API Documentation",
        version="1.0.0",
        description="API documentation for text and image processing using LLM",
        routes=app.routes,
    )

    # Add security scheme if needed later
    # openapi_schema["components"]["securitySchemes"] = { ... }

    # Custom documentation settings
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema
