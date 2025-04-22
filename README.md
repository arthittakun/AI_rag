# AI RAG API

## Setup Instructions

1. Create a `.env` file in the root directory with the following variables:
```
URL=your_api_endpoint_url
MODEL=your_model_name
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Make sure your `.env` file is properly configured
2. Run the FastAPI application:
```bash
python main.py
```

The API will be available at:
- API endpoint: http://localhost:8000
- Swagger documentation: http://localhost:8000/docs
- ReDoc documentation: http://localhost:8000/redoc

## API Endpoints

### Process Text
- Endpoint: `/process_text`
- Method: POST
- Body:
```json
{
    "prompt": "your text prompt"
}
```

### Process Image with Text
- Endpoint: `/process_image`
- Method: POST
- Body:
```json
{
    "prompt": "your text prompt",
    "image": "your image data"
}
```

