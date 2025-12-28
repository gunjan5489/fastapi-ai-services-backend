import os, json
import tempfile
from io import BytesIO
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException ,Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from PIL import Image
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from botocore import UNSIGNED
from botocore.config import Config
from typing import List, Optional
from google import genai  # pip install google-genai
import re
import logging
from auth import validate_api_key, check_permission, load_api_keys_from_env, APIKey
from rate_limiter import check_rate_limit

def strip_code_fences(text):
    # Remove ```json ... ``` or ``` ... ``` code fences
    return re.sub(r"^```(?:json)?\s*|```$", "", text.strip(), flags=re.MULTILINE).strip()

load_dotenv()  # Load environment variables from .env file
# Initialize authentication on startup
load_api_keys_from_env()
GEMINI_MODEL = os.getenv("GEMINI_MODEL")
GEMINI_MODEL_IMAGE = os.getenv("GEMINI_MODEL_IMAGE")

# Use GOOGLE_API_KEY in the environment per Google docs
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize S3 client
# AWS credentials can be set via environment variables:
# AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
# Or via IAM roles if running on AWS infrastructure
# For public buckets, we'll try anonymous access first
try:
    s3_client = boto3.client('s3')
except:
    s3_client = None

# Create anonymous S3 client for public buckets
s3_client_anonymous = boto3.client('s3', config=Config(signature_version=UNSIGNED))

# ----------------- S3 Helper Functions -----------------
def parse_s3_url(s3_url: str) -> tuple[str, str]:
    """
    Parse S3 URL to extract bucket and key.
    Supports both s3:// and https:// formats.

    Args:
        s3_url: S3 URL in format s3://bucket/key or https://bucket.s3.region.amazonaws.com/key

    Returns:
        Tuple of (bucket_name, object_key)
    """
    if s3_url.startswith('s3://'):
        # s3://bucket-name/path/to/object
        s3_url = s3_url[5:]  # Remove 's3://'
        parts = s3_url.split('/', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid S3 URL format: {s3_url}")
        return parts[0], parts[1]
    elif s3_url.startswith('https://'):
        # https://bucket-name.s3.region.amazonaws.com/path/to/object
        # or https://s3.region.amazonaws.com/bucket-name/path/to/object
        parsed = urlparse(s3_url)

        # Check if it's virtual-hosted-style URL
        if '.s3.' in parsed.netloc or '.s3-' in parsed.netloc:
            # Extract bucket name from subdomain
            bucket = parsed.netloc.split('.')[0]
            # Key is the path without leading slash
            key = parsed.path.lstrip('/')
        else:
            # Path-style URL
            path_parts = parsed.path.lstrip('/').split('/', 1)
            if len(path_parts) != 2:
                raise ValueError(f"Invalid S3 URL format: {s3_url}")
            bucket = path_parts[0]
            key = path_parts[1]

        return bucket, key
    else:
        raise ValueError(f"URL must start with 's3://' or 'https://': {s3_url}")

def download_s3_image(s3_url: str, use_anonymous: bool = True) -> Image.Image:
    """
    Download an image from S3 and return as PIL Image object.

    Args:
        s3_url: S3 URL of the image
        use_anonymous: Try anonymous access first for public buckets

    Returns:
        PIL Image object
    """
    try:
        bucket, key = parse_s3_url(s3_url)

        # Download image to BytesIO
        image_data = BytesIO()

        # Try anonymous access first for public buckets
        if use_anonymous:
            try:
                s3_client_anonymous.download_fileobj(bucket, key, image_data)
                image_data.seek(0)
                img = Image.open(image_data)
                return img
            except ClientError as e:
                # If anonymous access fails, try with credentials if available
                if s3_client and e.response['Error']['Code'] in ['403', 'AccessDenied']:
                    image_data = BytesIO()  # Reset the buffer
                    s3_client.download_fileobj(bucket, key, image_data)
                    image_data.seek(0)
                    img = Image.open(image_data)
                    return img
                else:
                    raise
        else:
            # Use authenticated client
            if not s3_client:
                raise HTTPException(500, "AWS credentials not configured and anonymous access disabled")
            s3_client.download_fileobj(bucket, key, image_data)
            image_data.seek(0)
            img = Image.open(image_data)
            return img

    except NoCredentialsError:
        # Try anonymous access as fallback
        try:
            image_data = BytesIO()
            s3_client_anonymous.download_fileobj(bucket, key, image_data)
            image_data.seek(0)
            img = Image.open(image_data)
            return img
        except:
            raise HTTPException(500, "AWS credentials not configured and public access failed")
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            raise HTTPException(404, f"S3 object not found: {s3_url}")
        elif error_code == '403':
            raise HTTPException(403, f"Access denied to S3 object: {s3_url}")
        else:
            raise HTTPException(500, f"S3 error: {e}")
    except Exception as e:
        raise HTTPException(400, f"Failed to download image from S3: {e}")

def download_s3_to_temp_file(s3_url: str) -> str:
    """
    Download an image from S3 to a temporary file and return the path.

    Args:
        s3_url: S3 URL of the image

    Returns:
        Path to temporary file
    """
    try:
        bucket, key = parse_s3_url(s3_url)

        # Create a temporary file
        # Get file extension from key
        _, ext = os.path.splitext(key)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)

        # Download to temp file
        s3_client.download_fileobj(bucket, key, temp_file)
        temp_file.close()

        return temp_file.name

    except NoCredentialsError:
        raise HTTPException(500, "AWS credentials not configured")
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            raise HTTPException(404, f"S3 object not found: {s3_url}")
        elif error_code == '403':
            raise HTTPException(403, f"Access denied to S3 object: {s3_url}")
        else:
            raise HTTPException(500, f"S3 error: {e}")
    except Exception as e:
        raise HTTPException(400, f"Failed to download image from S3: {e}")

# ----------------- SYSTEM PROMPTS -----------------
SYSTEM_PROMPT = """## Task
Analyze the provided `domx-document.json` file and Figma design image. ONLY identify DIV elements that
should be converted to semantic HTML5 tags and return a JSON mapping.

## Input Analysis Rules

### Step 1: Visual Section Identification
Look at the Figma design image for these specific visual markers:
- Clear horizontal divisions between content blocks
- Different background colors or visual styling
- Distinct content themes (navigation, projects, jobs, interviews, office info)
- Fixed positioning elements at the top of the page

### Step 2: JSON Node Analysis
For each potential semantic element, check:
- IF `metadata.figmaNode.name` contains words like "Header", "Container", "Section", "Navigation" → candidate for semantic tag
- IF `style.position` equals "fixed" AND `style.top` equals "0px" → likely HEADER
- IF element is a direct child of body AND contains multiple sub-elements → likely SECTION
- IF element serves only layout/styling purposes → keep as DIV

### Step 3: Decision Logic
Apply these rules in order:
1. IF element has fixed positioning at top → HEADER
2. IF element represents distinct thematic content visible as separate section in image → SECTION
3. IF element is purely for styling/layout → no change
4. ELSE → no change

## Concrete Example
Given a node with:
- `id: "i427"`
- `metadata.figmaNode.name: "Hero Section"`
- Contains hero text and background image
- Visually appears as distinct top section after header

Result: `"i427": {"name": "SECTION"}`

## Visual Reference Points
In the provided image, look for:
- Top navigation bar with logo (HEADER candidate)
- Large hero section with "DIGITAL INTERACTIVE DIV." text (SECTION candidate)
- "PROJECT" section with case studies (SECTION candidate)
- "JOB TYPE" section with role cards (SECTION candidate)
- "INTERVIEW" section with employee photos (SECTION candidate)
- "OFFICE" section at bottom (SECTION candidate)

## Output Requirements
ONLY return a JSON object with this exact structure. NO other text or explanation:

```json
{
  "node_id_1": {
    "name": "HEADER"
  },
  "node_id_2": {
    "name": "SECTION"
  }
}
```

## Critical Constraints
- ONLY analyze direct children of the body element
- ONLY change elements that clearly represent major page sections
- ONLY use HEADER, SECTION, or FOOTER tags
- ONLY include node IDs that need changes
- Return empty JSON `{}` if no changes needed
"""


TRANSLATE_PROMPT = """You are an expert localization specialist with deep multicultural competence and native-level fluency in multiple languages. You will receive a JSON array containing text from various digital platforms (websites, apps, marketing materials, technical documentation).

## CONTEXT AWARENESS
- **Source Languages**: Auto-detect from each text string (could be any language or mixed languages)
- **Content Types**: Dynamically identify: UI elements, marketing copy, technical documentation, legal text, creative content, user-generated content
- **Platform**: Digital content for web, mobile, or cross-platform use
- **Audience**: Adapt to implied audience based on content type and formality level

## CORE LOCALIZATION PRINCIPLES

### 1. INTELLIGENT DETECTION
- Auto-identify source language(s) for each text string
- Recognize mixed-language content and code-switching
- Detect content type from linguistic cues and formatting
- Identify cultural context markers
- **If the source text already matches the target language, exclude its node id from the final output (do not include untranslated items)**

### 2. ADAPTIVE TRANSLATION STRATEGY
Apply different approaches based on content type:

**UI/Navigation Elements**
- Maximum clarity and brevity
- Follow target platform conventions (web/mobile/desktop)
- Consistent terminology across interface

**Marketing/Creative Content**
- Preserve emotional impact and persuasive power
- Adapt cultural references and metaphors
- Maintain brand voice while ensuring local resonance

**Technical/Professional Content**
- Use established industry terminology
- Maintain precision while improving readability
- Adapt to local professional standards

**Legal/Compliance Text**
- Preserve legal accuracy
- Follow target jurisdiction conventions
- Maintain formal register

**User-Generated/Informal Content**
- Preserve tone and personality
- Adapt slang/colloquialisms appropriately
- Handle internet culture references

### 3. CULTURAL INTELLIGENCE
- Recognize and adapt idioms, metaphors, and cultural references
- Convert measurements, dates, currencies to local formats
- Adjust formality levels to target culture norms
- Handle humor, wordplay, and creative language appropriately

### 4. STRUCTURAL ADAPTATION
- Reorganize sentence structure for natural flow
- Adjust text direction for RTL languages when relevant
- Handle character limits for UI elements
- **Preserve all formatting markers exactly as they appear in the source text** (including `\n`, `\t`, `\r`, and other escape sequences)
- Maintain original spacing, line breaks, and special characters in their exact positions

### 5. SPECIAL HANDLING RULES

**Proper Names & Brands**
- Keep original for international brands
- Transliterate when necessary for script changes
- Add explanatory context if culturally specific

**Technical Terms & Acronyms**
- Use localized versions when established
- Keep international standards (ISO, IEEE, etc.)
- Provide clarification for region-specific terms

**Mixed Language Content**
- Identify intentional vs. incidental language mixing
- Preserve code-switching when stylistically important
- Unify language when mixing is incidental

**Untranslatable Concepts**
- Provide functional equivalents
- Use explanatory translation when needed
- Add cultural context notes in brackets when critical

### 6. QUALITY ASSURANCE CHECKLIST
For each translation, verify:
1. ✓ Natural and fluent in target language
2. ✓ Appropriate for detected content type
3. ✓ Culturally adapted, not just linguistically converted
4. ✓ Consistent with platform/medium requirements
5. ✓ Preserves original intent and impact
6. ✓ **All formatting characters (`\n`, `\t`, etc.) preserved in exact same positions as source**

## ERROR HANDLING
- For ambiguous content: Choose most likely interpretation based on context
- For unknown terms: Research or transliterate with explanation
- For cultural conflicts: Prioritize target audience appropriateness
- For technical limitations: Note if adaptation needed for character limits

## INPUT/OUTPUT SPECIFICATION
- **Input**: JSON array with objects containing `id` and `text` fields
- **Output**: Same structure with localized `text` values
- **Additional Optional Fields**:
  - `notes`: Critical context or warnings
  - `alternatives`: Other valid translations

## TARGET LANGUAGE: [Specify language and regional variant, e.g., "Portuguese (Brazil)" or "Spanish (Mexico)"]

## SPECIAL INSTRUCTIONS (if any):
[Space for client-specific requirements, glossaries, or style guides]

Do not return the entire DOM—only the nodes that had their "text" changed. Ensure each translation reads naturally as if originally written in the target language, not as a translation."""

IMAGE_ANALYSIS_PROMPT = """You are an expert cultural and marketing localization analyst. Your task is to analyze the provided image for its suitability for a specific target market. Analyze the image based on the following context and provide your output ONLY in JSON format.

**Context:**
- **Target Locale:** {target_locale}
- **Website Context:** {website_context}

**Analysis Schema (Your JSON output must follow this structure):**
{{
  "overallSuitabilityScore": "A score from 1-10 on how suitable the image is for the target locale.",
  "positiveElements": ["List of elements that are culturally appropriate or appealing."],
  "problematicElements": [
    {{
      "element": "Description of the element (e.g., 'Person's clothing', 'Text on sign', 'Type of food').",
      "reason": "Why this element might be unsuitable for the target locale (e.g., 'Clothing style is too informal for a premium brand in Japan', 'Text is in English', 'Food item is not common').",
      "suggestedChange": "A brief idea for a replacement."
    }}
  ],
  "textInImage": [
    {{
      "detectedText": "The exact text detected.",
      "language": "The detected language of the text."
    }}
  ],
  "generalAtmosphere": "Describe the overall mood and feeling of the image (e.g., 'Energetic and urban', 'Calm and natural')."
}}

Analyze the image provided."""

PROMPT_ENHANCEMENT_GUIDE = """You are an expert image generation prompt engineer specializing in creating culturally appropriate, localized visual content for international markets. Your task is to transform basic localization suggestions into detailed, actionable image generation prompts that will produce high-quality, culturally resonant imagery.

## Enhancement Framework

When receiving a localization suggestion, enhance it using this structured approach:

### 1. Subject Definition (Primary Focus)
- **People & Demographics**: Specify detailed demographic representation
  - Age ranges, ethnicities, professional attire
  - Natural body language and expressions appropriate to the culture
  - Group dynamics reflecting local social norms

### 2. Context & Environment
- **Setting Details**: Define the physical environment
  - Architecture style typical of the region
  - Environmental elements (weather, lighting, seasons)
  - Background elements that reinforce local authenticity

### 3. Style & Quality Modifiers
- **Visual Style**: Specify the photographic or artistic approach
  - Photography: "professional photograph", "4K", "HDR", "studio photo"
  - Lighting: "natural lighting", "warm office lighting", "golden hour"
  - Camera settings: "35mm lens", "shallow depth of field", "portrait"

### 4. Cultural Authenticity Markers
- **Local Elements**: Include region-specific details
  - Architectural styles common in the market
  - Technology and devices appropriate to local usage
  - Cultural artifacts or design elements (subtle, professional context)

## Enhanced Prompt Template

Transform the input suggestion into this format:

"A [quality modifier] [style] photograph of [detailed subject description] in [specific environment], featuring [cultural/demographic details], with [technical specifications]. The scene shows [action/interaction details] that reflects [cultural context]. [Composition details]. [Lighting and mood]. [Additional quality enhancers]."

## Key Enhancement Principles

### For Website/Marketing Context:
1. **Branding Authority**: Include modern design elements, high-end technology, professional settings
2. **Trust Building**: Show authentic interactions, genuine expressions, collaborative environments
3. **Cultural Resonance**: Feature locally recognizable elements without stereotypes
4. **Professional Quality**: Always specify high production values (4K, HDR, professional photography)

### Demographic Representation Guidelines:
- **Be Specific**: Rather than "diverse," specify actual ethnic backgrounds relevant to the market
- **Avoid Tokenism**: Show natural, integrated diversity as it exists in the target market
- **Professional Context**: Ensure all individuals appear as equals in professional capacity
- **Age Inclusivity**: Include various age groups when relevant (25-55 for professional settings)

### Technical Specifications to Always Include:
- Camera angle: "eye level", "slightly elevated", "dynamic angle"
- Lens type: "35mm", "50mm portrait", "24mm wide angle"
- Lighting: "natural lighting", "studio lighting", "golden hour"
- Quality markers: "4K", "high resolution", "professional photograph"
- Depth: "shallow depth of field", "bokeh background"

## Output Requirements

Given the suggestion and context, output ONLY the enhanced prompt that can be directly used for image generation. No explanations or additional text."""

USER_FRIENDLY_SUGGESTIONS_PROMPT = """You are a helpful creative assistant. Based on the following JSON analysis of an image, generate a concise, bulleted list of user-friendly suggestions for localizing this image. Frame the suggestions in a positive and constructive way.

**Analysis Data:**
{analysis_output}

**Target Locale:** {target_locale}

Provide your suggestions in a user-friendly format with bullet points. Start with:
"To better resonate with the {target_locale} audience, consider these enhancements:"

Then provide bullet points with categories like:
- **People:** [suggestions about models/people in the image]
- **Background:** [suggestions about scenery/environment]
- **Objects:** [suggestions about items/products shown]
- **Text:** [suggestions about any text in the image]
- **Atmosphere:** [suggestions about overall mood/feeling]

Only include categories that have relevant suggestions based on the analysis.
"""

# ----------------- Text Extraction Class -----------------
class DomxTextExtractor:
    """
    A utility class to extract all text nodes from a DOMX JSON structure.
    This prepares the data for efficient translation by an LLM.
    """
    @staticmethod
    def extract(domx_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Extracts all nodes with a non-empty 'text' property from the DOMX data.

        Args:
            domx_data: The parsed JSON content of the domx-document.json file.

        Returns:
            A list of dictionaries, where each dictionary contains the 'id' and 'text'
            of a node that has translatable content.
        """
        text_nodes = []
        if "nodes" not in domx_data or not isinstance(domx_data["nodes"], dict):
            return []
        for node_id, node_data in domx_data["nodes"].items():
            # Check if the node has a 'text' key with a non-whitespace value
            if "text" in node_data and isinstance(node_data["text"], str) and node_data["text"].strip():
                text_nodes.append({
                    "id": node_data.get("id", node_id),
                    "text": node_data["text"]
                })
        return text_nodes

# ----------------- FastAPI App Setup -----------------
app = FastAPI(title="AI Worker (FastAPI + Gemini + S3)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response validation

class SolveResponse(BaseModel):
    text: str

class ImageAnalysisRequest(BaseModel):
    target_locale: str = Form(..., description="Target locale/market (e.g., 'Japanese market', 'French audience')")
    website_context: str = Form(..., description="Context of the website (e.g., 'Premium fashion brand', 'Tech startup')")

class ImageAnalysisResponse(BaseModel):
    overallSuitabilityScore: str
    positiveElements: List[str]
    problematicElements: List[Dict[str, str]]
    textInImage: List[Dict[str, str]]
    generalAtmosphere: str

class LocalizationSuggestionsRequest(BaseModel):
    analysis_output: Dict[str, Any]
    target_locale: str

class LocalizationSuggestionsResponse(BaseModel):
    suggestions: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/v1/tags/resolve", response_model=SolveResponse)
async def solve(
    image_path: Optional[str] = Form(None),
    json_file: UploadFile = File(...),
    api_key: APIKey = Depends(check_rate_limit)  # Add rate limiting
):
    """
    Protected endpoint - requires valid API key
    """

    """
    Analyze a DOMX JSON document and optionally an image to suggest semantic HTML tag replacements.

    The image_path can be:
    - A local file path (absolute path visible to this container)
    - An S3 URL in format s3://bucket/key
    - An S3 HTTPS URL like https://bucket.s3.region.amazonaws.com/key
    """
    # Log API usage for monitoring
    print(f"API call by {api_key.name} ({api_key.environment})")

    try:
        raw_json_bytes = await json_file.read()
        json_text = raw_json_bytes.decode("utf-8")
        _ = json.loads(json_text)
    except Exception as e:
        raise HTTPException(400, f"Invalid JSON file: {e}")

    parts = [
        SYSTEM_PROMPT,
        f"Here is the JSON document to use:\n```json\n{json_text}\n```",
    ]

    temp_file_path = None

    if image_path:
        try:
            # Check if it's an S3 URL
            if image_path.startswith('s3://') or (image_path.startswith('https://') and ('s3.' in image_path or 's3-' in image_path)):
                # Download from S3
                img = download_s3_image(image_path)
                parts.append(img)
            else:
                # Treat as local file path
                if not os.path.exists(image_path):
                    raise HTTPException(400, f"Local image path not found: {image_path}")
                img = Image.open(image_path)
                parts.append(img)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(400, f"Failed to process image: {e}")

    try:
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=parts)
        cleaned_text = strip_code_fences(resp.text or "")
        return SolveResponse(text=cleaned_text or "")
    except Exception as e:
        raise HTTPException(500, f"Gemini API error: {e}")
    finally:
        # Clean up temp file if created
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass

@app.post("/v1/translate")
async def translate_domx(
    language: str = Form(..., description="Target language name, e.g., 'French', 'Japanese', 'Spanish'"),
    json_file: UploadFile = File(..., description="The domx-document.json file"),
     api_key: APIKey = Depends(check_permission("translate"))
):
    """
    Extracts text nodes, translates them using Gemini, and returns the translated nodes.
    """
    try:
        raw_json_bytes = await json_file.read()
        json_text = raw_json_bytes.decode("utf-8")
        domx_data = json.loads(json_text)
    except Exception as e:
        raise HTTPException(400, f"Invalid JSON file: {e}")

    # 1. Pre-process the DOMX JSON to extract only text nodes
    text_nodes_to_translate = DomxTextExtractor.extract(domx_data)

    if not text_nodes_to_translate:
        return {"translated_json": "[]"} # Return empty JSON array if no text found

    # Convert the extracted list to a JSON string for the prompt
    preprocessed_json_text = json.dumps(text_nodes_to_translate, ensure_ascii=False, indent=2)

    # 2. Build the prompt for the LLM with the simplified JSON
    parts = [
        TRANSLATE_PROMPT,
        f"**TARGET LANGUAGE:** \"{language}\"\n\n",
        f"**Input JSON Array:**\n```json\n{preprocessed_json_text}\n```"
    ]

    # 3. Call the Gemini API for translation
    try:
        resp_lang = client.models.generate_content(model=GEMINI_MODEL, contents=parts)
        cleaned_text = strip_code_fences(resp_lang.text or "")
        # Basic validation to ensure the response is valid JSON
        try:
            _ = json.loads(cleaned_text)
            return {"translated_json": cleaned_text}
        except json.JSONDecodeError:
            raise HTTPException(500, f"LLM returned invalid JSON: {cleaned_text}")
    except Exception as e:
        raise HTTPException(500, f"Gemini API error: {e}")

# Optional: Add endpoint to test S3 connectivity
@app.get("/v1/test-s3")
async def test_s3_connection():
    """
    Test if S3 credentials are configured and working.
    """
    try:
        # Try to list buckets as a simple connectivity test
        response = s3_client.list_buckets()
        return {
            "status": "connected",
            "buckets_count": len(response.get('Buckets', []))
        }
    except NoCredentialsError:
        return {
            "status": "error",
            "message": "AWS credentials not configured"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

# ----------------- New Multi-file Endpoints -----------------

@app.post("/v1/tags/resolve/multi", response_model=List[Dict])
async def solve_multi(
    json_files: List[UploadFile] = File(..., description="Multiple DOMX JSON files"),
    images: List[UploadFile] = File(default=[], description="Corresponding image files (optional, 1:1 with JSON files)"),
    image_paths: Optional[str] = Form(None, description="Comma-separated S3 URLs or local paths (optional, 1:1 with JSON files)"),
    api_key: APIKey = Depends(check_rate_limit)  # Add rate limiting
):
    """
    Analyze multiple DOMX JSON documents with their corresponding images (1:1 ratio).

    Each JSON file will be paired with its corresponding image:
    - json_files[0] pairs with images[0] or image_paths[0]
    - json_files[1] pairs with images[1] or image_paths[1]
    - etc.

    If no images are provided, processes JSON files without images.
    If number of images doesn't match number of JSON files, uses None for missing images.

    Returns a list of results, one for each JSON file.
    """
    results = []

    # Parse image paths if provided
    parsed_image_paths = []
    if image_paths:
        parsed_image_paths = [p.strip() for p in image_paths.split(',') if p.strip()]

    # Process each JSON file with its corresponding image
    for idx, json_file in enumerate(json_files):
        result = {
            "filename": json_file.filename,
            "index": idx,
            "result": None,
            "error": None,
            "image_source": None
        }

        # Read and validate JSON
        try:
            raw_json_bytes = await json_file.read()
            json_text = raw_json_bytes.decode("utf-8")
            _ = json.loads(json_text)
        except Exception as e:
            result["error"] = f"Invalid JSON file: {e}"
            results.append(result)
            continue

        # Prepare the base parts for Gemini
        parts = [
            SYSTEM_PROMPT,
            f"Here is the JSON document to use:\n```json\n{json_text}\n```",
        ]

        # Try to get corresponding image
        image_processed = False

        # First, check if there's an uploaded image at this index
        if idx < len(images):
            try:
                image_file = images[idx]
                image_bytes = await image_file.read()
                img = Image.open(BytesIO(image_bytes))
                parts.append(img)
                result["image_source"] = f"uploaded: {image_file.filename}"
                image_processed = True
            except Exception as e:
                result["error"] = f"Failed to process uploaded image: {e}"
                results.append(result)
                continue

        # If no uploaded image, check for image path at this index
        elif idx < len(parsed_image_paths):
            path = parsed_image_paths[idx]
            try:
                if path.startswith('s3://') or (path.startswith('https://') and ('s3.' in path or 's3-' in path)):
                    img = download_s3_image(path)
                    parts.append(img)
                    result["image_source"] = f"S3: {path}"
                else:
                    if not os.path.exists(path):
                        raise HTTPException(400, f"Local image path not found: {path}")
                    img = Image.open(path)
                    parts.append(img)
                    result["image_source"] = f"local: {path}"
                image_processed = True
            except HTTPException as he:
                result["error"] = str(he.detail)
                results.append(result)
                continue
            except Exception as e:
                result["error"] = f"Failed to process image path: {e}"
                results.append(result)
                continue

        # If no image available, process JSON only
        if not image_processed:
            result["image_source"] = "none"

        # Call Gemini API
        try:
            resp = client.models.generate_content(model=GEMINI_MODEL, contents=parts)
            cleaned_text = strip_code_fences(resp.text or "")
            result["result"] = cleaned_text or ""
        except Exception as e:
            result["error"] = f"Gemini API error: {e}"

        results.append(result)

    return results


@app.post("/v1/tags/resolve/multi/strict", response_model=List[Dict])
async def solve_multi_strict(
    json_files: List[UploadFile] = File(..., description="Multiple DOMX JSON files"),
    images: List[UploadFile] = File(..., description="Corresponding image files (must match count of JSON files)"),
    api_key: APIKey = Depends(check_rate_limit)  # Add rate limiting
):
    """
    Strict version: Requires exactly matching number of JSON files and images.
    Each JSON file must have a corresponding image file.

    Will return an error if the counts don't match.
    """
    # Validate matching counts
    if len(json_files) != len(images):
        raise HTTPException(
            400,
            f"Number of JSON files ({len(json_files)}) must match number of images ({len(images)})"
        )

    results = []

    # Process each JSON-image pair
    for idx, (json_file, image_file) in enumerate(zip(json_files, images)):
        result = {
            "json_filename": json_file.filename,
            "image_filename": image_file.filename,
            "index": idx,
            "result": None,
            "error": None
        }

        # Read and validate JSON
        try:
            raw_json_bytes = await json_file.read()
            json_text = raw_json_bytes.decode("utf-8")
            _ = json.loads(json_text)
        except Exception as e:
            result["error"] = f"Invalid JSON file: {e}"
            results.append(result)
            continue

        # Read and validate image
        try:
            image_bytes = await image_file.read()
            img = Image.open(BytesIO(image_bytes))
        except Exception as e:
            result["error"] = f"Invalid image file: {e}"
            results.append(result)
            continue

        # Prepare parts for Gemini
        parts = [
            SYSTEM_PROMPT,
            f"Here is the JSON document to use:\n```json\n{json_text}\n```",
            img
        ]

        # Call Gemini API
        try:
            resp = client.models.generate_content(model=GEMINI_MODEL, contents=parts)
            cleaned_text = strip_code_fences(resp.text or "")
            result["result"] = cleaned_text or ""
        except Exception as e:
            result["error"] = f"Gemini API error: {e}"

        results.append(result)

    return results

# Single file endpoint with direct upload (no S3/local path)
@app.post("/v1/tags/resolve/upload", response_model=SolveResponse)
async def solve_with_upload(
    json_file: UploadFile = File(..., description="DOMX JSON file"),
    image_file: Optional[UploadFile] = File(None, description="Image file (PNG, JPG, etc.)"),
    api_key: APIKey = Depends(check_permission("tags:resolve"))
):
    """
    Analyze a DOMX JSON document with direct image upload.

    This endpoint accepts the image as a file upload instead of requiring
    an S3 URL or local file path. This is useful when:
    - Working with images from client applications
    - Images are generated dynamically
    - You don't want to store images in S3
    - Development/testing without S3 setup
    """
    # Process JSON file
    try:
        raw_json_bytes = await json_file.read()
        json_text = raw_json_bytes.decode("utf-8")
        _ = json.loads(json_text)
    except Exception as e:
        raise HTTPException(400, f"Invalid JSON file: {e}")

    # Prepare parts for Gemini
    parts = [
        SYSTEM_PROMPT,
        f"Here is the JSON document to use:\n```json\n{json_text}\n```",
    ]

    # Process uploaded image if provided
    if image_file:
        try:
            # Read image bytes
            #print(type(image_file))
            image_bytes = await image_file.read()
            #print(type(image_bytes))
            # Validate it's a valid image
            img = Image.open(BytesIO(image_bytes))

            # Verify image format is supported
            if img.format not in ['PNG', 'JPEG', 'JPG', 'GIF', 'BMP', 'WEBP']:
                raise HTTPException(400, f"Unsupported image format: {img.format}")

            # Add image to parts for Gemini
            parts.append(img)

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(400, f"Failed to process uploaded image: {e}")

    # Call Gemini API
    try:
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=parts)
        cleaned_text = strip_code_fences(resp.text or "")
        return SolveResponse(text=cleaned_text or "")
    except Exception as e:
        raise HTTPException(500, f"Gemini API error: {e}")


# Batch upload endpoint for multiple files with direct upload
@app.post("/v1/tags/resolve/upload/batch", response_model=List[Dict])
async def solve_batch_upload(
    json_files: List[UploadFile] = File(..., description="Multiple DOMX JSON files"),
    image_files: List[UploadFile] = File(default=[], description="Corresponding image files (optional, 1:1 with JSON files)"),
    api_key: APIKey = Depends(check_rate_limit)
):
    """
    Batch processing with direct file uploads.

    Process multiple JSON-image pairs using direct file uploads.
    Images are optional - if provided, they should match the order of JSON files:
    - json_files[0] pairs with image_files[0]
    - json_files[1] pairs with image_files[1]
    - etc.

    This is useful for batch processing without needing S3 or local file storage.
    """
    results = []

    # Process each JSON file with its corresponding image
    for idx, json_file in enumerate(json_files):
        result = {
            "json_filename": json_file.filename,
            "index": idx,
            "result": None,
            "error": None,
            "has_image": False
        }

        # Read and validate JSON
        try:
            raw_json_bytes = await json_file.read()
            json_text = raw_json_bytes.decode("utf-8")
            _ = json.loads(json_text)
        except Exception as e:
            result["error"] = f"Invalid JSON file: {e}"
            results.append(result)
            continue

        # Prepare parts for Gemini
        parts = [
            SYSTEM_PROMPT,
            f"Here is the JSON document to use:\n```json\n{json_text}\n```",
        ]

        # Check if there's a corresponding image
        if idx < len(image_files):
            image_file = image_files[idx]
            result["image_filename"] = image_file.filename
            result["has_image"] = True

            try:
                # Read and validate image
                image_bytes = await image_file.read()
                img = Image.open(BytesIO(image_bytes))

                # Verify image format
                if img.format not in ['PNG', 'JPEG', 'JPG', 'GIF', 'BMP', 'WEBP']:
                    raise ValueError(f"Unsupported format: {img.format}")

                parts.append(img)

            except Exception as e:
                result["error"] = f"Failed to process image: {e}"
                results.append(result)
                continue

        # Call Gemini API
        try:
            resp = client.models.generate_content(model=GEMINI_MODEL, contents=parts)
            cleaned_text = strip_code_fences(resp.text or "")
            result["result"] = cleaned_text or ""
        except Exception as e:
            result["error"] = f"Gemini API error: {e}"

        results.append(result)

    return results




@app.post("/v1/translate/multi")
async def translate_domx_multi(
    languages: str = Form(..., description="Comma-separated target languages, e.g., 'French,Japanese,Spanish'"),
    json_files: List[UploadFile] = File(..., description="Multiple domx-document.json files"),
    api_key: APIKey = Depends(check_permission("translate"))
):
    """
    Translates multiple DOMX JSON files into multiple languages.
    Returns a dictionary with file names as keys and translation results as values.
    """
    target_languages = [lang.strip() for lang in languages.split(',') if lang.strip()]

    if not target_languages:
        raise HTTPException(400, "No target languages specified")

    results = {}

    for json_file in json_files:
        file_results = {}

        try:
            raw_json_bytes = await json_file.read()
            json_text = raw_json_bytes.decode("utf-8")
            domx_data = json.loads(json_text)
        except Exception as e:
            results[json_file.filename] = {"error": f"Invalid JSON file: {e}"}
            continue

        # Extract text nodes
        text_nodes_to_translate = DomxTextExtractor.extract(domx_data)

        if not text_nodes_to_translate:
            for lang in target_languages:
                file_results[lang] = "[]"
        else:
            preprocessed_json_text = json.dumps(text_nodes_to_translate, ensure_ascii=False, indent=2)

            # Translate to each language
            for language in target_languages:
                parts = [
                    TRANSLATE_PROMPT,
                    f"**Target Language:** \"{language}\"\n\n",
                    f"**Input JSON Array:**\n```json\n{preprocessed_json_text}\n```"
                ]

                try:
                    resp_lang = client.models.generate_content(model=GEMINI_MODEL, contents=parts)
                    cleaned_text = strip_code_fences(resp_lang.text or "")

                    # Validate JSON
                    try:
                        _ = json.loads(cleaned_text)
                        file_results[language] = cleaned_text
                    except json.JSONDecodeError:
                        file_results[language] = f"Error: Invalid JSON returned for {language}"
                except Exception as e:
                    file_results[language] = f"Error: {str(e)}"

        results[json_file.filename] = file_results

    return results


# ----------------- Combined Multi-operation Endpoint -----------------

class MultiOperationRequest(BaseModel):
    operations: List[str]  # ["resolve", "translate"]
    languages: Optional[List[str]] = None  # For translation

class MultiOperationResponse(BaseModel):
    filename: str
    resolve_results: Optional[List[Dict]] = None
    translation_results: Optional[Dict[str, str]] = None

@app.post("/v1/multi-operation", response_model=List[MultiOperationResponse])
async def multi_operation(
    operations: str = Form(..., description="Comma-separated operations: 'resolve,translate'"),
    languages: Optional[str] = Form(None, description="Comma-separated languages for translation"),
    json_files: List[UploadFile] = File(..., description="DOMX JSON files"),
    images: List[UploadFile] = File(default=[], description="Image files for resolve operation (1:1 with JSON files)"),
    image_paths: Optional[str] = Form(None, description="Comma-separated S3 URLs or paths (1:1 with JSON files)"),
    api_key: APIKey = Depends(check_rate_limit)
):
    """
    Perform multiple operations on multiple files in a single request.
    Operations can include 'resolve' and 'translate'.
    For resolve operation, images should be provided in 1:1 ratio with JSON files.
    """
    ops = [op.strip().lower() for op in operations.split(',') if op.strip()]
    target_languages = [lang.strip() for lang in languages.split(',')] if languages else []

    # Parse image paths
    parsed_image_paths = []
    if image_paths:
        parsed_image_paths = [p.strip() for p in image_paths.split(',') if p.strip()]

    results = []

    # Process each JSON file
    for idx, json_file in enumerate(json_files):
        result = MultiOperationResponse(filename=json_file.filename)

        try:
            raw_json_bytes = await json_file.read()
            json_text = raw_json_bytes.decode("utf-8")
            domx_data = json.loads(json_text)
        except Exception as e:
            result.resolve_results = [{"error": f"Invalid JSON: {e}"}]
            results.append(result)
            continue

        # Perform resolve operation
        if 'resolve' in ops:
            resolve_result = {"index": idx}

            parts = [
                SYSTEM_PROMPT,
                f"Here is the JSON document to use:\n```json\n{json_text}\n```",
            ]

            # Try to get corresponding image
            image_found = False

            # Check for uploaded image
            if idx < len(images):
                try:
                    image_file = images[idx]
                    image_bytes = await image_file.read()
                    img = Image.open(BytesIO(image_bytes))
                    parts.append(img)
                    resolve_result["image_source"] = f"uploaded: {image_file.filename}"
                    image_found = True
                except Exception as e:
                    resolve_result["error"] = f"Failed to process image: {e}"

            # Check for image path if no uploaded image
            elif idx < len(parsed_image_paths):
                path = parsed_image_paths[idx]
                try:
                    if path.startswith('s3://') or (path.startswith('https://') and ('s3.' in path or 's3-' in path)):
                        img = download_s3_image(path)
                    else:
                        img = Image.open(path)
                    parts.append(img)
                    resolve_result["image_source"] = f"path: {path}"
                    image_found = True
                except Exception as e:
                    resolve_result["error"] = f"Failed to process image path: {e}"

            if not image_found:
                resolve_result["image_source"] = "none"

            # Call Gemini API if no error
            if "error" not in resolve_result:
                try:
                    resp = client.models.generate_content(model=GEMINI_MODEL, contents=parts)
                    cleaned_text = strip_code_fences(resp.text or "")
                    resolve_result["result"] = cleaned_text
                except Exception as e:
                    resolve_result["error"] = str(e)

            result.resolve_results = [resolve_result]

        # Perform translate operation
        if 'translate' in ops and target_languages:
            translation_results = {}
            text_nodes = DomxTextExtractor.extract(domx_data)

            if text_nodes:
                preprocessed_json_text = json.dumps(text_nodes, ensure_ascii=False, indent=2)

                for language in target_languages:
                    parts = [
                        TRANSLATE_PROMPT,
                        f"**Target Language:** \"{language}\"\n\n",
                        f"**Input JSON Array:**\n```json\n{preprocessed_json_text}\n```"
                    ]

                    try:
                        resp = client.models.generate_content(model=GEMINI_MODEL, contents=parts)
                        cleaned_text = strip_code_fences(resp.text or "")
                        json.loads(cleaned_text)  # Validate
                        translation_results[language] = cleaned_text
                    except Exception as e:
                        translation_results[language] = f"Error: {str(e)}"
            else:
                for language in target_languages:
                    translation_results[language] = "[]"

            result.translation_results = translation_results

        results.append(result)

    return results


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/v1/generate-image/file")
async def generate_image_file(
    prompt: str = Form(..., description="Text prompt for image generation"),
    input_image: Optional[UploadFile] = File(None, description="Optional input image for editing/style transfer")
):
    """
    Generate or edit images and return as downloadable file.

    This endpoint returns the generated image as a direct file download
    instead of base64-encoded JSON response.
    High-cost endpoint - restricted permissions
    """
    if api_key.environment == "testing":
        raise HTTPException(
            status_code=403,
            detail="Image generation not available in testing environment"
        )
    try:
        # Log the received parameters for debugging
        logger.info(f"Received prompt: {prompt}")
        logger.info(f"Input image type: {type(input_image)}")

        # Prepare content parts for Gemini
        parts = [prompt]

        # Add input image if provided and valid
        if input_image and input_image.filename:  # Check if file actually exists
            logger.info(f"Processing input image: {input_image.filename}")
            logger.info(f"Content type: {input_image.content_type}")

            try:
                # Read the uploaded file
                image_bytes = await input_image.read()

                # Validate that we actually received image data
                if not image_bytes:
                    raise HTTPException(400, "Uploaded file is empty")

                # Open and validate the image
                img = Image.open(BytesIO(image_bytes))

                # Check supported formats
                if img.format not in ['PNG', 'JPEG', 'JPG', 'GIF', 'BMP', 'WEBP']:
                    raise HTTPException(400, f"Unsupported image format: {img.format}")

                # Convert image to appropriate format for Gemini if needed
                # Gemini typically expects PIL Image objects or bytes
                parts.append(img)

                logger.info(f"Successfully processed image: {img.format} {img.size}")

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to process input image: {e}")
                raise HTTPException(400, f"Failed to process input image: {str(e)}")

        # Call Gemini API
        logger.info("Calling Gemini API for image generation")
        print(GEMINI_MODEL_IMAGE)
        response = client.models.generate_content(
            model=GEMINI_MODEL_IMAGE,
            contents=parts
        )

        # Extract generated image
        generated_image_data = None

        # Check if response has candidates
        if not response.candidates:
            raise HTTPException(500, "No candidates in generation response")

        # Extract image data from response
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'inline_data') and part.inline_data is not None:
                generated_image_data = part.inline_data.data
                break

        if generated_image_data is None:
            raise HTTPException(500, "No image was generated in the response")

        # Return image as streaming response
        return StreamingResponse(
            BytesIO(generated_image_data),
            media_type="image/png",
            headers={
                "Content-Disposition": "attachment; filename=generated_image.png",
                "Content-Type": "image/png"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        raise HTTPException(500, f"Image generation failed: {str(e)}")


@app.post("/v1/image/enhance-prompt")
async def enhance_localization_prompt(
    suggestion: str = Form(..., description="The localization suggestion to enhance"),
    target_locale: str = Form(..., description="Target locale/market"),
    website_context: str = Form(..., description="Context of the website"),
    api_key: APIKey = Depends(check_permission("image:analyze"))
):
    """
    Enhance a basic localization suggestion into a detailed image generation prompt.

    This endpoint takes simple suggestions like "make the team more diverse" and
    transforms them into comprehensive, actionable prompts for image generation.
    """

    # Build the enhancement request
    enhancement_request = f"""{PROMPT_ENHANCEMENT_GUIDE}

**Input Suggestion:**
{suggestion}

**Target Locale:** {target_locale}
**Website Context:** {website_context}

Transform this suggestion into a detailed, actionable image generation prompt following the enhancement framework."""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[enhancement_request]
        )

        enhanced_prompt = response.text or ""

        return {
            "original_suggestion": suggestion,
            "enhanced_prompt": enhanced_prompt,
            "target_locale": target_locale,
            "website_context": website_context
        }

    except Exception as e:
        raise HTTPException(500, f"Gemini API error: {str(e)}")

@app.post("/v1/image/full-localization-pipeline")
async def full_localization_pipeline(
    target_locale: str = Form(..., description="Target locale/market"),
    website_context: str = Form(..., description="Context of the website"),
    original_image: Optional[UploadFile] = File(None, description="Original image to analyze and localize"),
    original_image_path: Optional[str] = Form(None, description="S3 URL or local path to original image"),
    custom_generation_prompt: Optional[str] = Form(None, description="Custom prompt for final image generation"),
    auto_generate: bool = Form(True, description="Automatically generate new image based on analysis"),
    api_key: APIKey = Depends(check_permission("image:generate"))
):
    """
    Complete localization pipeline: Analyze → Suggest → Generate new image.

    This endpoint:
    1. Analyzes the original image for the target locale
    2. Generates localization suggestions
    3. (Optionally) Generates a new localized image

    Returns analysis, suggestions, and the generated image (if auto_generate=True).
    """

    # Validate image source
    if not original_image and not original_image_path:
        raise HTTPException(400, "Either 'original_image' file upload or 'original_image_path' must be provided")

    # Process the original image
    img_to_analyze = None

    if original_image and original_image.filename:
        try:
            image_bytes = await original_image.read()
            if not image_bytes:
                raise HTTPException(400, "Uploaded file is empty")

            # Reset the file pointer so we can read it again later if needed
            await original_image.seek(0)

            img_to_analyze = Image.open(BytesIO(image_bytes))

            if img_to_analyze.format not in ['PNG', 'JPEG', 'JPG', 'GIF', 'BMP', 'WEBP']:
                raise HTTPException(400, f"Unsupported image format: {img_to_analyze.format}")

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(400, f"Failed to process uploaded image: {str(e)}")

    elif original_image_path:
        try:
            if original_image_path.startswith('s3://') or (original_image_path.startswith('https://') and ('s3.' in original_image_path or 's3-' in original_image_path)):
                img_to_analyze = download_s3_image(original_image_path)
            else:
                if not os.path.exists(original_image_path):
                    raise HTTPException(400, f"Local image path not found: {original_image_path}")
                img_to_analyze = Image.open(original_image_path)

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(400, f"Failed to process image: {str(e)}")

    # Step 1: Analyze the image
    analysis_prompt = IMAGE_ANALYSIS_PROMPT.format(
        target_locale=target_locale,
        website_context=website_context
    )

    try:
        analysis_response = client.models.generate_content(
            model=GEMINI_MODEL_IMAGE,
            contents=[analysis_prompt, img_to_analyze]
        )

        cleaned_analysis = strip_code_fences(analysis_response.text or "")
        analysis_result = json.loads(cleaned_analysis)

    except json.JSONDecodeError:
        raise HTTPException(500, f"Model returned invalid JSON for analysis: {cleaned_analysis}")
    except Exception as e:
        raise HTTPException(500, f"Gemini API error during analysis: {str(e)}")

    # Step 2: Generate suggestions
    suggestions_prompt = USER_FRIENDLY_SUGGESTIONS_PROMPT.format(
        analysis_output=json.dumps(analysis_result, indent=2),
        target_locale=target_locale
    )

    try:
        suggestions_response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[suggestions_prompt]
        )

        suggestions_text = suggestions_response.text or ""

    except Exception as e:
        raise HTTPException(500, f"Gemini API error during suggestions generation: {str(e)}")

    # Prepare response
    response_data = {
        "analysis": analysis_result,
        "suggestions": suggestions_text,
        "target_locale": target_locale,
        "website_context": website_context
    }

    # Step 3: Generate new image if requested

    if auto_generate:
        # Build generation prompt
        if custom_generation_prompt and custom_generation_prompt != "string":
            generation_prompt = custom_generation_prompt
        else:
            # First, enhance the suggestions into a detailed prompt
            base_suggestion = f"Use this image as a style reference, adapting it for the target market: {suggestions_text}"

            enhancement_request = f"""{PROMPT_ENHANCEMENT_GUIDE}

                                    **Input Suggestion:**
                                    {base_suggestion}

                                    **Target Locale:** {target_locale}
                                    **Website Context:** {website_context}

                                    Based on these localization recommendations:
                                    {suggestions_text}

                                    Key improvements needed:"""

            # Add specific improvements from analysis
            if analysis_result.get("problematicElements"):
                for element in analysis_result["problematicElements"][:5]:
                    if element.get("suggestedChange"):
                        enhancement_request += f"\n- {element['suggestedChange']}"

            enhancement_request += "\n\nTransform this into a detailed, actionable image generation prompt."

            try:
                # Get enhanced prompt
                enhance_response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[enhancement_request]
                )
                generation_prompt = enhance_response.text or ""
                logger.info(f"Enhanced prompt: {generation_prompt}")
            except Exception as e:
                logger.warning(f"Failed to enhance prompt: {e}, using fallback")
                # Fallback to basic prompt
                generation_prompt = (
                    f"Generate a culturally appropriate image for {target_locale} market.\n"
                    f"Website context: {website_context}\n\n"
                    f"Based on these localization recommendations:\n"
                    f"{suggestions_text}\n\n"
                    f"Create a professional, high-quality image that addresses these suggestions."
                )


            logger.info(generation_prompt)



            # Prepare parts for generation
            parts = [generation_prompt]
            #logger.info(parts)
            # Use original image as style reference
            parts.insert(0, "Use this image as a style reference, adapting it for the target market:")
            parts.append(img_to_analyze)
            logger.info(parts)
            try:
                logger.info("Generating localized image")
                gen_response = client.models.generate_content(
                model=GEMINI_MODEL_IMAGE,
                contents=parts
                # contents=[
                #     generation_prompt,
                #     img_to_analyze
                #     ]
                )

                # Extract generated image
                generated_image_data = None

                if gen_response.candidates:
                    for part in gen_response.candidates[0].content.parts:
                        if hasattr(part, 'inline_data') and part.inline_data is not None:
                            generated_image_data = part.inline_data.data
                            break

                if generated_image_data:
                    # Create filename with context
                    filename = f"localized_{target_locale.replace(' ', '_').lower()}_generated.png"

                    # Return the generated image as a downloadable file
                    return StreamingResponse(
                        BytesIO(generated_image_data),
                        media_type="image/png",
                        headers={
                            "Content-Disposition": f"attachment; filename={filename}",
                            "Content-Type": "image/png",
                            "X-Analysis-Score": str(analysis_result.get("overallSuitabilityScore", "N/A")),
                            "X-Target-Locale": target_locale,
                            "X-Website-Context": website_context
                        }
                    )
                else:
                    response_data["generated_image_available"] = False
                    response_data["generation_error"] = "No image was generated"

            except Exception as e:
                logger.error(f"Failed to generate localized image: {e}")
                response_data["generated_image_available"] = False
                response_data["generation_error"] = str(e)

    # If no image was generated or auto_generate was False, return JSON response
    return response_data