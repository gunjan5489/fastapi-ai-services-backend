# AI Worker - AI-Powered DOM Analysis & Localization API

A comprehensive FastAPI service that leverages Google Gemini models to provide intelligent DOM analysis, translation, and image localization capabilities. Features API key authentication and rate limiting for production use.

## 🚀 Features

### Core Capabilities

- **Semantic HTML Tag Resolution**: Analyzes DOM structures with Figma designs to suggest semantic HTML improvements
- **Multi-Language Translation**: Translates webpage content into multiple target languages with cultural adaptation
- **Image Localization Analysis**: Evaluates images for cultural appropriateness in target markets
- **AI Image Generation**: Creates culturally adapted images based on localization analysis
- **Batch Processing**: Handle multiple files and operations simultaneously
- **S3 Integration**: Direct support for AWS S3 image storage and retrieval

### Security & Management

- **API Key Authentication**: Secure access with environment-based API keys
- **Rate Limiting**: Built-in rate limiting per API key and environment
- **Permissions System**: Granular endpoint permissions per API key
- **Daily Rotating Logs**: Automatic log rotation with timestamp-based files

## 📋 Prerequisites

- Python 3.8+
- Google API Key with Gemini API access ([Get one here](https://aistudio.google.com/app/apikey))
- (Optional) AWS credentials for S3 integration
- (Optional) Docker for containerized deployment

## 🗂️ Project Structure

```
ai-worker/
├── src/
│   ├── .env.development          # Development environment variables
│   ├── .env.testing              # Testing environment variables
│   ├── app.py                    # Main FastAPI application
│   ├── auth.py                   # Authentication & API key management
│   ├── client.py                 # Client utilities
│   ├── docker-compose.yml        # Main Docker Compose config
│   ├── docker-compose.dev.yml    # Development Docker config
│   ├── docker-compose.prod.yml   # Production Docker config
│   ├── docker-compose.test.yml   # Testing Docker config
│   ├── Dockerfile                # Docker image configuration
│   ├── env.example               # Environment variables template
│   ├── key_generator.py          # API key generation utility
│   ├── rate_limiter.py           # Rate limiting implementation
│   ├── README.md                 # This file
│   └── requirements.txt          # Python dependencies
```

## 🛠️ Setup and Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ai-worker/src
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

#### Basic Setup

```bash
cp env.example .env
```

Edit `.env` and add your credentials:

```env
# Google Gemini API Configuration
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_MODEL=gemini-2.0-flash-exp
GEMINI_MODEL_IMAGE=gemini-2.0-flash-exp

# AWS S3 Configuration (Optional)
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_DEFAULT_REGION=us-east-1

# API Authentication
API_KEYS=key1:name1:env1:perm1,perm2;key2:name2:env2:perm3
# Format: api_key:name:environment:comma_separated_permissions

# Example API key configuration:
# API_KEYS=sk-prod-abc123:ProductionApp:production:all;sk-dev-xyz789:DevApp:development:translate,tags:resolve;sk-test-qwe456:TestClient:testing:translate
```

#### Generate API Keys

Use the included key generator utility:

```bash
python key_generator.py
```

This will generate a secure API key that you can add to your environment configuration.

### 5. Environment-Specific Configuration

The application supports three environments:
- **Production** (`.env`)
- **Development** (`.env.development`)
- **Testing** (`.env.testing`)

Each environment can have different API keys, rate limits, and permissions.

## 🏃 Running the Application

### Local Development

```bash
# Using default environment
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Using specific environment
export ENV=development
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Using Docker

#### Build and Run with Docker

```bash
# Build the image
docker build -t ai-worker:latest .

# Run with environment file
docker run -d --name ai-worker -p 8000:8000 --env-file .env ai-worker:latest
```

#### Using Docker Compose

```bash
# Development environment
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Production environment
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Testing environment
docker-compose -f docker-compose.yml -f docker-compose.test.yml up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 🔐 Authentication & Rate Limiting

### API Key Format

API keys are configured in the environment with the following format:
```
api_key:name:environment:permission1,permission2
```

### Permissions

Available permissions include:
- `all` - Access to all endpoints
- `translate` - Access to translation endpoints
- `tags:resolve` - Access to tag resolution endpoints
- `image:analyze` - Access to image analysis endpoints
- `image:generate` - Access to image generation endpoints

### Rate Limiting

Rate limits are applied per API key and can be configured by environment:
- Production: 100 requests per minute (default)
- Development: 200 requests per minute (default)
- Testing: 50 requests per minute (default)

### Using API Keys in Requests

Include your API key in the request headers:

```bash
curl -X POST http://localhost:8000/v1/translate \
  -H "X-API-Key: your-api-key-here" \
  -F "json_file=@sample.json" \
  -F "language=Japanese"
```

## 📚 API Endpoints

### Health Check

```http
GET /health
```

Returns service status (no authentication required)

### 1. Tag Resolution Endpoints

#### Single File Analysis

```http
POST /v1/tags/resolve
```

- **Headers**: `X-API-Key: your-key`
- **Parameters**:
  - `json_file`: DOMX JSON file (required)
  - `image_path`: S3 URL or local path (optional)
- **Returns**: Suggested semantic HTML tag mappings

#### Direct Upload

```http
POST /v1/tags/resolve/upload
```

- **Headers**: `X-API-Key: your-key`
- **Parameters**:
  - `json_file`: DOMX JSON file (required)
  - `image_file`: Image file upload (optional)

#### Batch Processing

```http
POST /v1/tags/resolve/multi
```

- **Headers**: `X-API-Key: your-key`
- **Parameters**:
  - `json_files[]`: Multiple DOMX JSON files
  - `images[]`: Corresponding images (optional, 1:1 ratio)
  - `image_paths`: Comma-separated S3 URLs

### 2. Translation Endpoints

#### Single Language Translation

```http
POST /v1/translate
```

- **Headers**: `X-API-Key: your-key`
- **Parameters**:
  - `json_file`: DOMX JSON file
  - `language`: Target language (e.g., "Japanese", "Spanish")
- **Returns**: Translated text nodes

#### Multi-Language Translation

```http
POST /v1/translate/multi
```

- **Headers**: `X-API-Key: your-key`
- **Parameters**:
  - `json_files[]`: Multiple DOMX JSON files
  - `languages`: Comma-separated target languages

### 3. Image Localization Endpoints

#### Generate Localized Image

```http
POST /v1/generate-image/file
```

- **Headers**: `X-API-Key: your-key`
- **Parameters**:
  - `prompt`: Text description for image generation
  - `input_image`: Optional base image for style reference
- **Returns**: Generated image file

#### Full Localization Pipeline

```http
POST /v1/image/full-localization-pipeline
```

- **Headers**: `X-API-Key: your-key`
- **Parameters**:
  - `target_locale`: Target market
  - `website_context`: Website context
  - `original_image`: Image to analyze and localize
  - `auto_generate`: Generate new image (default: true)
- **Returns**: Analysis, suggestions, and optionally generated image

## 📊 Logging & Monitoring

### Log Files

Logs are automatically saved to the `logs/` directory with daily rotation:
```
logs/
├── log_2024-01-15.txt
├── log_2024-01-16.txt
└── log_2024-01-17.txt
```

### View Logs

```bash
# Docker logs
docker logs ai-worker -f

# Local log files
tail -f logs/log_$(date +%Y-%m-%d).txt
```

### API Documentation

When running, access interactive API documentation at:
- Swagger UI: `http://localhost:8000/ai/docs`
- ReDoc: `http://localhost:8000/ai/redoc`

## 🧪 Testing

### Quick Test

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test with API key
curl -X POST http://localhost:8000/v1/tags/resolve \
  -H "X-API-Key: your-api-key" \
  -F "json_file=@sample_domx.json" \
  -F "image_path=s3://bucket/image.png"

# Test translation
curl -X POST http://localhost:8000/v1/translate \
  -H "X-API-Key: your-api-key" \
  -F "json_file=@sample_domx.json" \
  -F "language=Japanese"
```

### Run Test Suite

```bash
pytest tests/ -v
```

## 📧 S3 Configuration

The service supports both public and private S3 buckets:

### Public Buckets

No configuration needed - the service attempts anonymous access by default

### Private Buckets

Set AWS credentials via:
1. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
2. IAM roles (when running on AWS infrastructure)
3. AWS CLI configuration

### Supported S3 URL Formats

- `s3://bucket-name/path/to/object`
- `https://bucket-name.s3.region.amazonaws.com/path/to/object`
- `https://s3.region.amazonaws.com/bucket-name/path/to/object`

## 🏗️ Architecture

```
┌─────────────┐       ┌──────────────┐     ┌─────────────┐
│   Client    │───▶   │  FastAPI     │───▶│  Gemini     │
│(Streamlit)  │◀───   │   Server     │◀───│    API      │
└─────────────┘       └──────────────┘     └─────────────┘
                           │    │
                           │    └──────────▶ ┌─────────────┐
                           │                 │    Auth     │
                           │                 │   System    │
                           │                 └─────────────┘
                           ▼
                    ┌──────────────┐
                    │   AWS S3     │
                    │  (Optional)  │
                    └──────────────┘
```

## 🔒 Security Considerations

- **API Keys**: Store securely using environment variables or secrets management
- **Rate Limiting**: Automatically enforced per API key
- **Environment Isolation**: Separate keys and limits for dev/test/prod
- **CORS**: Configure appropriately for production (currently allows all origins)
- **File Uploads**: Implement size limits and file type validation
- **Logging**: Sensitive data is not logged; logs rotate daily

## 🚀 Performance Tips

1. **Batch Operations**: Use multi-file endpoints for better throughput
2. **Image Optimization**: Compress images before uploading
3. **Caching**: Consider implementing Redis for frequently accessed data
4. **Token Usage**: Monitor Gemini token consumption to optimize costs
5. **Async Processing**: For large batches, consider implementing job queues
6. **Rate Limit Planning**: Adjust rate limits based on your usage patterns

## 🆘 Troubleshooting

### Common Issues

1. **Authentication Failed**
   - Check API key is correctly formatted in environment
   - Verify key has required permissions

2. **Rate Limited**
   - Wait for rate limit window to reset (1 minute)
   - Consider upgrading to production environment

3. **S3 Access Denied**
   - Verify AWS credentials are configured
   - Check bucket permissions

4. **Gemini API Errors**
   - Verify GOOGLE_API_KEY is valid
   - Check API quotas and limits
