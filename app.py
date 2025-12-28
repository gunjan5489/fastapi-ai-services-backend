import os, json
import tempfile
from io import BytesIO
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
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
from fastapi.openapi.utils import get_openapi
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
import time
import traceback
from auth import validate_api_key, check_permission, load_api_keys_from_env, APIKey
from rate_limiter import check_rate_limit
from google.genai import types
# ===================== LOGGING CONFIGURATION =====================
def setup_logging():
    """
    Configure logging with daily rotating file handler
    Creates log files with format: log_YYYY-MM-DD.txt
    """
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Get today's date for the log filename
    today = datetime.now().strftime("%Y-%m-%d")
    log_filename = os.path.join(log_dir, f"log_{today}.txt")

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Create and configure file handler with daily rotation
    file_handler = TimedRotatingFileHandler(
        filename=log_filename,
        when='midnight',  # Rotate at midnight
        interval=1,       # Every 1 day
        backupCount=30,   # Keep 30 days of logs
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # Console handler for development
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return logging.getLogger(__name__)

# Initialize logging
logger = setup_logging()
logger.info("=" * 80)
logger.info("APPLICATION STARTING UP")
logger.info("=" * 80)

def strip_code_fences(text):
    # Remove ```json ... ``` or ``` ... ``` code fences
    return re.sub(r"^```(?:json)?\s*|```$", "", text.strip(), flags=re.MULTILINE).strip()

def get_standard_aspect_ratio(width, height):
    """Return the closest standard aspect ratio from the supported list"""
    ratio = width / height

    # All supported aspect ratios from Gemini Image API
    standard_ratios = {
        "1:1": 1.0,           # Square
        "2:3": 2/3,           # Portrait
        "3:2": 3/2,           # Landscape
        "3:4": 3/4,           # Portrait
        "4:3": 4/3,           # Landscape
        "4:5": 4/5,           # Portrait
        "5:4": 5/4,           # Landscape
        "9:16": 9/16,         # Portrait (mobile)
        "16:9": 16/9,         # Landscape (widescreen)
        "21:9": 21/9          # Landscape (ultra-wide)
    }

    # Find closest match
    closest_ratio = min(standard_ratios.items(),
                       key=lambda x: abs(x[1] - ratio))

    # Log the difference for debugging
    difference = abs(standard_ratios[closest_ratio[0]] - ratio)
    if difference > 0.1:  # If difference is significant
        logger.warning(f"Image ratio {ratio:.3f} differs significantly from closest standard {closest_ratio[0]}")

    return closest_ratio[0]
# Load environment variables
load_dotenv()
logger.info("Environment variables loaded from .env file")

# Initialize authentication on startup
load_api_keys_from_env()
logger.info("API keys loaded from environment")

GEMINI_MODEL = os.getenv("GEMINI_MODEL")
GEMINI_MODEL_IMAGE = os.getenv("GEMINI_MODEL_IMAGE")
logger.info(f"Gemini models configured - Text: {GEMINI_MODEL}, Image: {GEMINI_MODEL_IMAGE}")

# Use GOOGLE_API_KEY in the environment per Google docs
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
logger.info("Google Gemini client initialized")

# Initialize S3 client
try:
    s3_client = boto3.client('s3')
    logger.info("AWS S3 client initialized successfully")
except Exception as e:
    s3_client = None
    logger.warning(f"Failed to initialize AWS S3 client: {e}")

# Create anonymous S3 client for public buckets
try:
    s3_client_anonymous = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    logger.info("Anonymous S3 client initialized for public buckets")
except Exception as e:
    logger.error(f"Failed to initialize anonymous S3 client: {e}")
    s3_client_anonymous = None

# ----------------- S3 Helper Functions -----------------
def parse_s3_url(s3_url: str) -> tuple[str, str]:
    """
    Parse S3 URL to extract bucket and key.
    Supports both s3:// and https:// formats.
    """
    logger.debug(f"Parsing S3 URL: {s3_url}")

    if s3_url.startswith('s3://'):
        s3_url = s3_url[5:]
        parts = s3_url.split('/', 1)
        if len(parts) != 2:
            logger.error(f"Invalid S3 URL format: {s3_url}")
            raise ValueError(f"Invalid S3 URL format: {s3_url}")
        bucket, key = parts[0], parts[1]
        logger.debug(f"Parsed S3 URL - Bucket: {bucket}, Key: {key}")
        return bucket, key
    elif s3_url.startswith('https://'):
        parsed = urlparse(s3_url)

        if '.s3.' in parsed.netloc or '.s3-' in parsed.netloc:
            bucket = parsed.netloc.split('.')[0]
            key = parsed.path.lstrip('/')
        else:
            path_parts = parsed.path.lstrip('/').split('/', 1)
            if len(path_parts) != 2:
                logger.error(f"Invalid S3 URL format: {s3_url}")
                raise ValueError(f"Invalid S3 URL format: {s3_url}")
            bucket = path_parts[0]
            key = path_parts[1]

        logger.debug(f"Parsed HTTPS S3 URL - Bucket: {bucket}, Key: {key}")
        return bucket, key
    else:
        logger.error(f"URL must start with 's3://' or 'https://': {s3_url}")
        raise ValueError(f"URL must start with 's3://' or 'https://': {s3_url}")

def download_s3_image(s3_url: str, use_anonymous: bool = True) -> Image.Image:
    """
    Download an image from S3 and return as PIL Image object.
    """
    start_time = time.time()
    logger.info(f"Downloading S3 image: {s3_url} (anonymous: {use_anonymous})")

    try:
        bucket, key = parse_s3_url(s3_url)
        image_data = BytesIO()

        if use_anonymous:
            try:
                logger.debug(f"Attempting anonymous download from bucket: {bucket}, key: {key}")
                s3_client_anonymous.download_fileobj(bucket, key, image_data)
                image_data.seek(0)
                img = Image.open(image_data)
                elapsed = time.time() - start_time
                logger.info(f"Successfully downloaded image via anonymous access in {elapsed:.2f}s")
                return img
            except ClientError as e:
                error_code = e.response['Error']['Code']
                logger.warning(f"Anonymous access failed with error {error_code}, trying authenticated access")
                if s3_client and error_code in ['403', 'AccessDenied']:
                    image_data = BytesIO()
                    s3_client.download_fileobj(bucket, key, image_data)
                    image_data.seek(0)
                    img = Image.open(image_data)
                    elapsed = time.time() - start_time
                    logger.info(f"Successfully downloaded image via authenticated access in {elapsed:.2f}s")
                    return img
                else:
                    raise
        else:
            if not s3_client:
                logger.error("AWS credentials not configured and anonymous access disabled")
                raise HTTPException(500, "AWS credentials not configured and anonymous access disabled")
            s3_client.download_fileobj(bucket, key, image_data)
            image_data.seek(0)
            img = Image.open(image_data)
            elapsed = time.time() - start_time
            logger.info(f"Successfully downloaded image via authenticated access in {elapsed:.2f}s")
            return img

    except NoCredentialsError:
        logger.warning("No AWS credentials found, attempting anonymous access")
        try:
            image_data = BytesIO()
            s3_client_anonymous.download_fileobj(bucket, key, image_data)
            image_data.seek(0)
            img = Image.open(image_data)
            elapsed = time.time() - start_time
            logger.info(f"Successfully downloaded image via anonymous fallback in {elapsed:.2f}s")
            return img
        except Exception as e:
            logger.error(f"Anonymous access fallback failed: {e}")
            raise HTTPException(500, "AWS credentials not configured and public access failed")
    except ClientError as e:
        error_code = e.response['Error']['Code']
        logger.error(f"S3 ClientError - Code: {error_code}, Message: {e}")
        if error_code == '404':
            raise HTTPException(404, f"S3 object not found: {s3_url}")
        elif error_code == '403':
            raise HTTPException(403, f"Access denied to S3 object: {s3_url}")
        else:
            raise HTTPException(500, f"S3 error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error downloading S3 image: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(400, f"Failed to download image from S3: {e}")

def download_s3_to_temp_file(s3_url: str) -> str:
    """
    Download an image from S3 to a temporary file and return the path.
    """
    logger.info(f"Downloading S3 file to temp: {s3_url}")

    try:
        bucket, key = parse_s3_url(s3_url)
        _, ext = os.path.splitext(key)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        logger.debug(f"Created temp file: {temp_file.name}")

        s3_client.download_fileobj(bucket, key, temp_file)
        temp_file.close()

        logger.info(f"Successfully downloaded to temp file: {temp_file.name}")
        return temp_file.name

    except NoCredentialsError:
        logger.error("AWS credentials not configured")
        raise HTTPException(500, "AWS credentials not configured")
    except ClientError as e:
        error_code = e.response['Error']['Code']
        logger.error(f"S3 ClientError - Code: {error_code}")
        if error_code == '404':
            raise HTTPException(404, f"S3 object not found: {s3_url}")
        elif error_code == '403':
            raise HTTPException(403, f"Access denied to S3 object: {s3_url}")
        else:
            raise HTTPException(500, f"S3 error: {e}")
    except Exception as e:
        logger.error(f"Failed to download S3 file to temp: {e}")
        raise HTTPException(400, f"Failed to download image from S3: {e}")

# ----------------- SYSTEM PROMPTS -----------------

SYSTEM_PROMPT = """
## Task
Analyze the provided `domx-document.json` file and Figma design image. ONLY identify DIV elements that
should be converted to semantic HTML5 tags and return a JSON mapping.

## Input Analysis Rules

### Step 1: Visual Section Identification
Look at the Figma design image for these specific visual markers:
- Clear horizontal divisions between content blocks
- Different background colors or visual styling
- Distinct content themes (navigation, main content, articles, sidebars, footers)
- Fixed positioning elements at the top or bottom of the page
- Navigation menus and link groups
- Self-contained content blocks (blog posts, cards, product listings)
- Sidebar or complementary content
- Images with captions

### Step 2: JSON Node Analysis
For each potential semantic element, check the `metadata.figmaNode.name` for keywords:
- "Header", "Top Bar" → HEADER
- "Footer", "Bottom" → FOOTER
- "Navigation", "Nav", "Menu" → NAV
- "Main", "Content", "Primary" → MAIN
- "Section", "Container", "Block" → SECTION
- "Article", "Post", "Card", "Item" → ARTICLE
- "Sidebar", "Aside", "Related" → ASIDE
- "Figure", "Image Container" + caption → FIGURE

### Step 3: Decision Logic
Apply these rules in order:

1. **HEADER**: Fixed positioning at top OR top-level element with logo/branding/site title
2. **NAV**: Contains navigation links, menu items, or site navigation structure
3. **MAIN**: Primary content area (should be unique per page)
4. **ARTICLE**: Self-contained, independently distributable content (blog post, news article, product card, user comment)
5. **SECTION**: Thematic grouping of content with a clear heading or topic
6. **ASIDE**: Tangentially related content (sidebar, pull quotes, advertisements, related links)
7. **FOOTER**: Bottom-level element with copyright, links, contact info, social media
8. **FIGURE**: Contains image/video with associated caption (look for figcaption or caption text nearby)
9. **DIV**: Purely for styling/layout purposes with no semantic meaning

### Step 4: Hierarchy Rules
- A page should have only ONE `<main>` element
- `<article>` can contain `<section>` elements
- `<section>` should typically have a heading
- `<header>` and `<footer>` can exist within `<article>` or `<section>`, not just at page level
- `<nav>` should contain navigation links (consider parent-child relationships)

## Concrete Examples

### Example 1: Header
```json
{
  "id": "i101",
  "metadata.figmaNode.name": "Top Navigation",
  "style.position": "fixed",
  "style.top": "0px"
}
Result: "i101": {"name": "HEADER"}
Example 2: Navigation
json{
  "id": "i202",
  "metadata.figmaNode.name": "Main Menu",
  "contains": ["Home", "About", "Services", "Contact"]
}
Result: "i202": {"name": "NAV"}
Example 3: Article Card
json{
  "id": "i303",
  "metadata.figmaNode.name": "Project Card",
  "contains": ["title", "description", "image", "date"]
}
Result: "i303": {"name": "ARTICLE"}
Example 4: Sidebar
json{
  "id": "i404",
  "metadata.figmaNode.name": "Related Links Sidebar"
}
Result: "i404": {"name": "ASIDE"}
Visual Reference Points
In the provided image, identify:

HEADER: Top navigation bar with logo and main menu
NAV: Menu items, link groups
MAIN: Primary content wrapper (if there's one main content area)
SECTION: Distinct content blocks (Hero, Projects, Job Types, Interviews, Office)
ARTICLE: Individual project cards, job listings, employee profiles
ASIDE: Sidebar content, related information panels
FOOTER: Bottom section with contact info, copyright, social links
FIGURE: Images with captions

Output Requirements
ONLY return a JSON object with this exact structure. NO other text or explanation:
json{
  "node_id_1": {
    "name": "HEADER"
  },
  "node_id_2": {
    "name": "NAV"
  },
  "node_id_3": {
    "name": "MAIN"
  },
  "node_id_4": {
    "name": "SECTION"
  },
  "node_id_5": {
    "name": "ARTICLE"
  },
  "node_id_6": {
    "name": "ASIDE"
  },
  "node_id_7": {
    "name": "FOOTER"
  },
  "node_id_8": {
    "name": "FIGURE"
  }
}
Supported Semantic Tags

HEADER: Site/section header
NAV: Navigation links
MAIN: Main content (use once per page)
SECTION: Thematic content section
ARTICLE: Self-contained content
ASIDE: Sidebar/tangential content
FOOTER: Site/section footer
FIGURE: Image/media with caption

Critical Constraints

Prioritize semantic meaning over visual appearance
Consider content purpose, not just layout
ONLY change elements that have clear semantic meaning
ONLY include node IDs that need changes
Return empty JSON {} if no changes needed
When in doubt, keep as DIV
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
1. ✔ Natural and fluent in target language
2. ✔ Appropriate for detected content type
3. ✔ Culturally adapted, not just linguistically converted
4. ✔ Consistent with platform/medium requirements
5. ✔ Preserves original intent and impact
6. ✔ **All formatting characters (`\n`, `\t`, etc.) preserved in exact same positions as source**

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


PROMPT_ENHANCEMENT_GUIDE = """You are an expert image generation prompt engineer specializing in creating culturally appropriate, localized visual content for international markets, optimized specifically for Gemini 2.5 Flash Image model capabilities and limitations.

## Critical Model-Specific Considerations

### Gemini 2.5 Flash Image Limitations to Address:
1. **TEXT GENERATION**: Model cannot reliably generate readable text - avoid requesting specific text
2. **ANATOMICAL CONSISTENCY**: Hands and eyes require careful prompting to avoid artifacts
   - **HAND ARTIFACT HOTSPOTS**: Gloves, mittens, holding small objects, fingers in complex positions
   - **HIGH-RISK**: Any request involving detailed hand accessories or finger articulation
3. **CULTURAL ACCURACY**: Requires explicit, detailed cultural markers
4. **REPETITION DEGRADATION**: Same prompt generates artifacts - build variation into prompts

## Enhancement Framework

When receiving a localization suggestion, enhance it using this structured approach:

### 1. Subject Definition (Primary Focus)
- **People & Demographics**: Specify detailed demographic representation
  - Age ranges, ethnicities, professional attire
  - **HANDS MANAGEMENT - CRITICAL**: Use artifact-prevention strategies
    - ✅ SAFE: "hands resting naturally on desk", "arms crossed", "hands at sides"
    - ✅ SAFE: "hand resting on railing", "hands clasped together"
    - ⚠️ MODERATE RISK: "holding simple mug with both hands" (specify "with relaxed grip, minimal finger detail visible")
    - ❌ HIGH RISK: "wearing gloves/mittens", "holding small objects showing finger detail", "gesturing with visible fingers"
  - **When holding objects IS necessary**: Use these safer approaches:
    - "holding [object] with hands partially out of frame"
    - "holding [object] with hands slightly blurred"
    - "gripping [object] with simple hand position, fingers naturally curled"
    - Avoid: gloves, mittens, detailed finger positions, multiple small objects
  - **EYE CONTACT**: "making natural eye contact with camera", "looking at colleague", "eyes focused forward"
  - Natural body language and expressions appropriate to the culture
  - Group dynamics reflecting local social norms (max 3-4 people to reduce complexity)

### 2. Context & Environment
- **Setting Details**: Define the physical environment
  - Architecture style typical of the region with specific landmarks or styles
  - Environmental elements (weather, lighting, seasons)
  - Background elements that reinforce local authenticity
  - **AVOID TEXT REQUESTS**: NO signs, documents, screens with readable text, branded items requiring text
  - **SAFE ALTERNATIVES**: "blank presentation screen", "minimalist laptop display", "abstract charts", "blurred background signage"

### 3. Style & Quality Modifiers
- **Visual Style**: Specify the photographic or artistic approach
  - Photography: "professional photograph", "sharp focus", "high detail", "studio photo"
  - Lighting: "soft natural lighting", "even studio lighting", "warm office lighting", "diffused light"
  - Camera settings: "50mm portrait lens", "medium shot", "slight bokeh"
  - **AVOID**: Extreme close-ups of hands/faces, complex angles that expose hand details

### 4. Cultural Authenticity Markers
- **Local Elements**: Include region-specific details with precision
  - Named architectural styles: "modern Korean office building", "traditional Japanese minimalist interior"
  - Specific clothing details: "wearing a light blue barong tagalog", "in modern business hanbok style"
  - Regional color preferences: "warm earth tones common in Indonesian design", "bright colors typical of Latin American aesthetics"
  - Technology appropriate to market: "latest smartphone model", "modern laptop"
  - Food/beverage elements when relevant: "traditional tea service", "local coffee presentation"
  - **BE SPECIFIC**: Replace vague terms like "Asian" with precise ethnicities

## Enhanced Prompt Template

Transform the input suggestion into this format:

"A professional, high-detail photograph of [detailed subject with specific ethnicity and age] in [precisely described local environment], [explicit safe hand position], [specific eye direction/expression], wearing [detailed culturally appropriate clothing]. The [specific local architectural/design element] is visible in the background. Shot with a 50mm lens, soft natural lighting, slight bokeh effect, medium shot composition. The scene conveys [specific emotional context] appropriate to [cultural context]. No text visible. Sharp focus on faces, professional photography, authentic [specific market] setting."

## Key Enhancement Principles

### For Website/Marketing Context:
1. **Branding Authority**: Include modern design elements, high-end technology, professional settings
2. **Trust Building**: Show authentic interactions, genuine expressions, collaborative environments
3. **Cultural Resonance**: Name specific elements (e.g., "Seoul's modern architecture", "traditional Filipino patterns")
4. **Professional Quality**: Always specify high production values but keep compositions simple

### Demographic Representation Guidelines:
- **Be Precise**: Specify exact ethnic backgrounds: "Korean businesswoman, age 35", "Filipino male entrepreneur, age 40"
- **Avoid Tokenism**: Show natural, integrated diversity as it exists in the target market
- **Professional Context**: Ensure all individuals appear as equals in professional capacity
- **Age Inclusivity**: Include specific age ranges (30-45 for professional settings)
- **Limit Complexity**: Maximum 3-4 people per image to reduce artifact risk

### Anatomical Artifact Prevention:
**CRITICAL - HAND POSITION HIERARCHY (Safest to Riskiest):**

**TIER 1 - SAFEST (Always prefer these):**
- ✅ "arms crossed professionally at chest"
- ✅ "hands resting flat on desk surface"
- ✅ "hands clasped together in lap"
- ✅ "one hand resting on railing, other at side"
- ✅ "hands relaxed at sides"
- ✅ "arms folded comfortably"

**TIER 2 - MODERATE (Use with caution, add safety modifiers):**
- ⚠️ "holding simple mug with both hands, relaxed grip, minimal finger detail"
- ⚠️ "hand resting on laptop edge" (not typing)
- ⚠️ "holding tablet flat against body"
- **Safety modifiers**: "hands partially visible", "fingers naturally relaxed", "simple grip"

**TIER 3 - HIGH RISK (AVOID):**
- ❌ "wearing gloves or mittens" (very high artifact rate)
- ❌ "holding pen/phone showing finger detail"
- ❌ "typing on keyboard"
- ❌ "gesturing with hands"
- ❌ "pointing"
- ❌ "holding multiple small objects"
- ❌ "interlaced fingers in detail"
- ❌ "hand accessories (rings, watches) in focus"

**WINTER/OUTDOOR SCENE EXCEPTION:**
If the scene MUST include winter/cold weather where gloves would be natural:
- Use: "hands in pockets of winter coat"
- Use: "holding warm drink with hands mostly obscured by sleeves"
- Use: "arms wrapped around body for warmth"
- NEVER: "wearing detailed gloves/mittens holding objects"

**EYE GUIDELINES:**
- ✅ "looking directly at camera with confident, warm expression"
- ✅ "making eye contact with colleague, genuine smile"
- ✅ "eyes focused on middle distance, thoughtful expression"
- ❌ Avoid: "looking down at phone", extreme side-eye, unusual angles

### Text Handling Strategies:
**NEVER request readable text. Use these alternatives:**
- Instead of "computer screen with code": → "laptop with abstract blue interface glow"
- Instead of "whiteboard with notes": → "modern whiteboard with colorful abstract diagrams"
- Instead of "business signage": → "blurred professional signage in background"
- Instead of "documents": → "minimal paperwork, text out of focus"
- Instead of "branded materials": → "sleek modern materials without visible branding"
- Instead of "name tags/badges": → "professional attire without visible badges"

### Technical Specifications to Always Include:
- Composition: "medium shot", "waist-up framing", "portrait orientation"
- Lens type: "50mm portrait lens" (most reliable), "35mm lens" (for groups)
- Lighting: "soft natural lighting", "even studio lighting", "diffused daylight"
- Quality markers: "high detail", "sharp focus on faces", "professional photograph"
- Depth: "slight bokeh background" (not too extreme)
- Focus strategy: "sharp focus on faces, hands softly rendered in background"
- **AVOID**: "macro", "extreme close-up", "wide angle below 24mm", "dramatic hand gestures"

### Prompt Variation Techniques:
**To prevent repetition-based artifacts when generating multiple images:**
- Vary lighting descriptions: "morning light" vs "afternoon glow" vs "soft overcast lighting"
- Alternate composition slightly: "centered composition" vs "rule of thirds"
- Change minor environmental details: "potted plant visible" vs "modern art piece in background"
- Adjust clothing colors: "navy blue business attire" vs "charcoal gray professional wear"
- Modify background depth: "shallow depth of field" vs "moderate background blur"
- Rotate hand positions through TIER 1 options
- Vary facial directions: "looking at camera" vs "looking slightly left" vs "engaged with colleague"

### Cultural Authenticity Checklist:
For each market, verify inclusion of:
- ✅ Specific ethnicity (not just "Asian" or "Latino")
- ✅ Named architectural or design style
- ✅ Appropriate professional attire for that market
- ✅ Culturally appropriate business interaction style
- ✅ Regional color palette preferences
- ✅ Local environmental context (climate, urban/nature balance)
- ✅ Season-appropriate clothing without high-risk hand accessories

## Example Enhancements

**Basic Input**: "Show Indian couple enjoying winter vacation in mountains"

**WRONG Enhancement** (would produce artifacts like the reference image):
"Indian couple wearing winter gloves and beanies, holding steaming cups of chai on snowy mountain balcony..."

**CORRECT Enhancement**:
"A professional, high-detail photograph of an Indian couple (man age 32, woman age 29) on a wooden balcony with dramatic snow-covered Himalayan peaks in soft-focus background, during gentle snowfall. The man wearing navy blue winter jacket and charcoal knit beanie, the woman wearing burgundy winter jacket and cream knit beanie with their long dark hair visible. Both have arms wrapped around each other for warmth, their hands not prominently visible, tucked naturally into the embrace. They are looking up and to the left with genuine smiles and warm expressions, eyes reflecting joy. Traditional alpine-style chalet architecture with wooden details visible behind them. Shot with 50mm portrait lens, soft diffused winter daylight, slight bokeh on mountain background, medium shot from waist up. Gentle snowflakes in foreground, warm interior lighting glow from chalet windows. The scene conveys romantic winter getaway authentic to Indian couples' travel preferences. Sharp focus on faces, natural expressions, professional photography. No visible text, no detailed hand positions, no gloves holding objects."

**Basic Input**: "Show diverse team in Japanese office"

**Enhanced Output**: "A professional, high-detail photograph of a Japanese businessman (age 38) and businesswoman (age 34) in a modern Tokyo office with minimalist design and natural wood elements, both wearing contemporary business attire in navy and gray tones. The man has his arms crossed professionally at chest height, the woman has her hands resting flat on the glass meeting table surface. Both making natural eye contact with camera with confident, welcoming expressions and subtle smiles. Large floor-to-ceiling windows show Tokyo's modern skyline with characteristic high-rise buildings slightly blurred in background. Sleek closed laptop with subtle blue standby light on table, no readable text or screens. Shot with 50mm portrait lens, soft natural afternoon lighting from windows, slight bokeh effect, medium shot showing subjects from waist up. The scene conveys professional collaboration in authentic Japanese corporate environment with characteristic attention to clean aesthetics. Sharp focus on faces, high detail in clothing texture, professional photography, minimal color palette typical of Japanese business settings."

## Output Requirements

Given the suggestion and context, output ONLY the enhanced prompt that can be directly used for image generation. The prompt must:
1. Use ONLY Tier 1 (safest) hand positions, or explicitly avoid showing hands in detail
2. NEVER request gloves, mittens, or hand accessories when hands are visible/active
3. Contain NO requests for readable text
4. Include specific cultural/ethnic details
5. Specify technical parameters that reduce artifacts
6. Include "sharp focus on faces" while keeping hands less detailed
7. Be detailed enough to prevent repetition issues
8. Be a single, flowing prompt (not bullet points)

Focus on clear, natural descriptions that work with the model's strengths while carefully avoiding its documented weaknesses, especially the critical hand-artifact issue with gloves and detailed finger positions."""

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
    """
    @staticmethod
    def extract(domx_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Extracts all nodes with a non-empty 'text' property from the DOMX data.
        """
        logger.debug("Extracting text nodes from DOMX data")
        text_nodes = []

        if "nodes" not in domx_data or not isinstance(domx_data["nodes"], dict):
            logger.warning("No nodes found in DOMX data")
            return []

        for node_id, node_data in domx_data["nodes"].items():
            if "text" in node_data and isinstance(node_data["text"], str) and node_data["text"].strip():
                text_nodes.append({
                    "id": node_data.get("id", node_id),
                    "text": node_data["text"]
                })

        logger.debug(f"Extracted {len(text_nodes)} text nodes")
        return text_nodes

# ----------------- FastAPI App Setup -----------------
APP_VERSION = "1.0.0"
app = FastAPI(
    title="AI Worker (FastAPI + Gemini + S3)",
    version=APP_VERSION,
    root_path="/ai",  # Important for path prefix
)

# Custom OpenAPI schema with server configuration
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="AI Worker (FastAPI + Gemini + S3)",
        version=APP_VERSION,
        description="AI Service API for DOMX processing, translation, and image generation",
        routes=app.routes,
    )
    openapi_schema["servers"] = [
        {"url": "/ai", "description": "Production server"}
    ]
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
logger.info("FastAPI app initialized")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("CORS middleware configured")

# Pydantic models for request/response validation
class SolveResponse(BaseModel):
    text: str

class ImageAnalysisRequest(BaseModel):
    target_locale: str = Form(..., description="Target locale/market")
    website_context: str = Form(..., description="Context of the website")

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
    logger.debug("Health check endpoint called")
    return {"status": "ok"}

@app.post("/v1/tags/resolve", response_model=SolveResponse)
async def solve(
    image_path: Optional[str] = Form(None),
    json_file: UploadFile = File(...),
    api_key: APIKey = Depends(check_rate_limit)
):
    """
    Protected endpoint - requires valid API key
    """
    start_time = time.time()
    request_id = f"resolve_{int(time.time() * 1000)}"

    logger.info(f"[{request_id}] Tags resolve request from {api_key.name} ({api_key.environment})")
    logger.info(f"[{request_id}] JSON file: {json_file.filename}, Image path: {image_path}")

    try:
        raw_json_bytes = await json_file.read()
        json_text = raw_json_bytes.decode("utf-8")
        _ = json.loads(json_text)
        logger.debug(f"[{request_id}] Successfully parsed JSON file ({len(json_text)} bytes)")
    except Exception as e:
        logger.error(f"[{request_id}] Invalid JSON file: {e}")
        raise HTTPException(400, f"Invalid JSON file: {e}")

    parts = [
        SYSTEM_PROMPT,
        f"Here is the JSON document to use:\n```json\n{json_text}\n```",
    ]

    temp_file_path = None

    if image_path:
        try:
            if image_path.startswith('s3://') or (image_path.startswith('https://') and ('s3.' in image_path or 's3-' in image_path)):
                logger.info(f"[{request_id}] Downloading image from S3: {image_path}")
                img = download_s3_image(image_path)
                parts.append(img)
            else:
                if not os.path.exists(image_path):
                    logger.error(f"[{request_id}] Local image not found: {image_path}")
                    raise HTTPException(400, f"Local image path not found: {image_path}")
                logger.info(f"[{request_id}] Loading local image: {image_path}")
                img = Image.open(image_path)
                parts.append(img)
            logger.info(f"[{request_id}] Image loaded successfully")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[{request_id}] Failed to process image: {e}")
            raise HTTPException(400, f"Failed to process image: {e}")

    try:
        logger.info(f"[{request_id}] Calling Gemini API")
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=parts)
        cleaned_text = strip_code_fences(resp.text or "")

        elapsed = time.time() - start_time
        logger.info(f"[{request_id}] Request completed successfully in {elapsed:.2f}s")

        return SolveResponse(text=cleaned_text or "")
    except Exception as e:
        logger.error(f"[{request_id}] Gemini API error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(500, f"Gemini API error: {e}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.debug(f"[{request_id}] Cleaned up temp file: {temp_file_path}")
            except:
                pass

@app.post("/v1/translate")
async def translate_domx(
    language: str = Form(..., description="Target language name"),
    json_file: UploadFile = File(..., description="The domx-document.json file"),
    api_key: APIKey = Depends(check_permission("translate"))
):
    """
    Extracts text nodes, translates them using Gemini, and returns the translated nodes.
    """
    start_time = time.time()
    request_id = f"translate_{int(time.time() * 1000)}"

    logger.info(f"[{request_id}] Translate request from {api_key.name} to language: {language}")
    logger.info(f"[{request_id}] JSON file: {json_file.filename}")

    try:
        raw_json_bytes = await json_file.read()
        json_text = raw_json_bytes.decode("utf-8")
        domx_data = json.loads(json_text)
        logger.debug(f"[{request_id}] Successfully parsed JSON file")
    except Exception as e:
        logger.error(f"[{request_id}] Invalid JSON file: {e}")
        raise HTTPException(400, f"Invalid JSON file: {e}")

    text_nodes_to_translate = DomxTextExtractor.extract(domx_data)
    logger.info(f"[{request_id}] Extracted {len(text_nodes_to_translate)} text nodes for translation")

    if not text_nodes_to_translate:
        logger.info(f"[{request_id}] No text nodes found to translate")
        return {"translated_json": "[]"}

    preprocessed_json_text = json.dumps(text_nodes_to_translate, ensure_ascii=False, indent=2)

    parts = [
        TRANSLATE_PROMPT,
        f"**TARGET LANGUAGE:** \"{language}\"\n\n",
        f"**Input JSON Array:**\n```json\n{preprocessed_json_text}\n```"
    ]

    try:
        logger.info(f"[{request_id}] Calling Gemini API for translation")
        resp_lang = client.models.generate_content(model=GEMINI_MODEL, contents=parts)
        cleaned_text = strip_code_fences(resp_lang.text or "")

        try:
            _ = json.loads(cleaned_text)
            elapsed = time.time() - start_time
            logger.info(f"[{request_id}] Translation completed successfully in {elapsed:.2f}s")
            return {"translated_json": cleaned_text}
        except json.JSONDecodeError:
            logger.error(f"[{request_id}] LLM returned invalid JSON")
            raise HTTPException(500, f"LLM returned invalid JSON: {cleaned_text}")
    except Exception as e:
        logger.error(f"[{request_id}] Gemini API error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(500, f"Gemini API error: {e}")

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
    start_time = time.time()
    request_id = f"translate_multi_{int(time.time() * 1000)}"

    target_languages = [lang.strip() for lang in languages.split(',') if lang.strip()]

    logger.info(f"[{request_id}] Multi-translate request from {api_key.name} ({api_key.environment})")
    logger.info(f"[{request_id}] Processing {len(json_files)} files for {len(target_languages)} languages: {', '.join(target_languages)}")

    if not target_languages:
        logger.error(f"[{request_id}] No target languages specified")
        raise HTTPException(400, "No target languages specified")

    results = {}
    total_translations = 0
    successful_translations = 0
    failed_translations = 0

    for file_idx, json_file in enumerate(json_files):
        file_start_time = time.time()
        file_request_id = f"{request_id}_file{file_idx}"

        logger.info(f"[{file_request_id}] Processing file {file_idx + 1}/{len(json_files)}: {json_file.filename}")
        file_results = {}

        try:
            raw_json_bytes = await json_file.read()
            json_text = raw_json_bytes.decode("utf-8")
            domx_data = json.loads(json_text)
            logger.debug(f"[{file_request_id}] Successfully parsed JSON file ({len(json_text)} bytes)")
        except Exception as e:
            logger.error(f"[{file_request_id}] Invalid JSON file: {e}")
            results[json_file.filename] = {"error": f"Invalid JSON file: {e}"}
            failed_translations += len(target_languages)
            continue

        # Extract text nodes
        text_nodes_to_translate = DomxTextExtractor.extract(domx_data)
        logger.info(f"[{file_request_id}] Extracted {len(text_nodes_to_translate)} text nodes")

        if not text_nodes_to_translate:
            logger.info(f"[{file_request_id}] No text nodes found to translate")
            for lang in target_languages:
                file_results[lang] = "[]"
                successful_translations += 1
        else:
            preprocessed_json_text = json.dumps(text_nodes_to_translate, ensure_ascii=False, indent=2)

            # Translate to each language
            for lang_idx, language in enumerate(target_languages):
                lang_start_time = time.time()
                lang_request_id = f"{file_request_id}_lang{lang_idx}"

                logger.info(f"[{lang_request_id}] Translating to {language} ({lang_idx + 1}/{len(target_languages)})")
                total_translations += 1

                parts = [
                    TRANSLATE_PROMPT,
                    f"**Target Language:** \"{language}\"\n\n",
                    f"**Input JSON Array:**\n```json\n{preprocessed_json_text}\n```"
                ]

                try:
                    logger.debug(f"[{lang_request_id}] Calling Gemini API")
                    resp_lang = client.models.generate_content(model=GEMINI_MODEL, contents=parts)
                    cleaned_text = strip_code_fences(resp_lang.text or "")

                    # Validate JSON
                    try:
                        validated_json = json.loads(cleaned_text)
                        file_results[language] = cleaned_text
                        successful_translations += 1

                        lang_elapsed = time.time() - lang_start_time
                        logger.info(f"[{lang_request_id}] Translation successful in {lang_elapsed:.2f}s")

                    except json.JSONDecodeError as json_err:
                        error_msg = f"Invalid JSON returned for {language}"
                        logger.error(f"[{lang_request_id}] {error_msg}: {json_err}")
                        logger.debug(f"[{lang_request_id}] Raw response: {cleaned_text[:500]}...")
                        file_results[language] = f"Error: {error_msg}"
                        failed_translations += 1

                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"[{lang_request_id}] Gemini API error: {error_msg}")
                    logger.error(traceback.format_exc())
                    file_results[language] = f"Error: {error_msg}"
                    failed_translations += 1

        file_elapsed = time.time() - file_start_time
        logger.info(f"[{file_request_id}] File processed in {file_elapsed:.2f}s")
        results[json_file.filename] = file_results

    total_elapsed = time.time() - start_time
    success_rate = (successful_translations / total_translations * 100) if total_translations > 0 else 0

    logger.info(f"[{request_id}] ========== TRANSLATION SUMMARY ==========")
    logger.info(f"[{request_id}] Total files processed: {len(json_files)}")
    logger.info(f"[{request_id}] Total languages: {len(target_languages)}")
    logger.info(f"[{request_id}] Total translation attempts: {total_translations}")
    logger.info(f"[{request_id}] Successful translations: {successful_translations}")
    logger.info(f"[{request_id}] Failed translations: {failed_translations}")
    logger.info(f"[{request_id}] Success rate: {success_rate:.1f}%")
    logger.info(f"[{request_id}] Total processing time: {total_elapsed:.2f}s")
    logger.info(f"[{request_id}] Average time per translation: {(total_elapsed/total_translations if total_translations > 0 else 0):.2f}s")
    logger.info(f"[{request_id}] =========================================")

    return results

@app.get("/v1/test-s3")
async def test_s3_connection():
    """
    Test if S3 credentials are configured and working.
    """
    logger.info("Testing S3 connection")

    try:
        response = s3_client.list_buckets()
        bucket_count = len(response.get('Buckets', []))
        logger.info(f"S3 connection successful, found {bucket_count} buckets")
        return {
            "status": "connected",
            "buckets_count": bucket_count
        }
    except NoCredentialsError:
        logger.error("AWS credentials not configured")
        return {
            "status": "error",
            "message": "AWS credentials not configured"
        }
    except Exception as e:
        logger.error(f"S3 connection test failed: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/v1/tags/resolve/multi", response_model=List[Dict])
async def solve_multi(
    json_files: List[UploadFile] = File(...),
    images: List[UploadFile] = File(default=[]),
    image_paths: Optional[str] = Form(None),
    api_key: APIKey = Depends(check_rate_limit)
):
    """
    Analyze multiple DOMX JSON documents with their corresponding images (1:1 ratio).
    """
    start_time = time.time()
    request_id = f"resolve_multi_{int(time.time() * 1000)}"

    logger.info(f"[{request_id}] Multi-resolve request from {api_key.name}")
    logger.info(f"[{request_id}] Processing {len(json_files)} JSON files, {len(images)} images")

    results = []

    parsed_image_paths = []
    if image_paths:
        parsed_image_paths = [p.strip() for p in image_paths.split(',') if p.strip()]
        logger.debug(f"[{request_id}] Parsed {len(parsed_image_paths)} image paths")

    for idx, json_file in enumerate(json_files):
        file_start_time = time.time()
        file_request_id = f"{request_id}_file{idx}"

        logger.info(f"[{file_request_id}] Processing file {idx + 1}/{len(json_files)}: {json_file.filename}")

        result = {
            "filename": json_file.filename,
            "index": idx,
            "result": None,
            "error": None,
            "image_source": None
        }

        try:
            raw_json_bytes = await json_file.read()
            json_text = raw_json_bytes.decode("utf-8")
            _ = json.loads(json_text)
            logger.debug(f"[{file_request_id}] Successfully parsed JSON")
        except Exception as e:
            logger.error(f"[{file_request_id}] Invalid JSON: {e}")
            result["error"] = f"Invalid JSON file: {e}"
            results.append(result)
            continue

        parts = [
            SYSTEM_PROMPT,
            f"Here is the JSON document to use:\n```json\n{json_text}\n```",
        ]

        image_processed = False

        if idx < len(images):
            try:
                image_file = images[idx]
                logger.debug(f"[{file_request_id}] Processing uploaded image: {image_file.filename}")
                image_bytes = await image_file.read()
                img = Image.open(BytesIO(image_bytes))
                parts.append(img)
                result["image_source"] = f"uploaded: {image_file.filename}"
                image_processed = True
            except Exception as e:
                logger.error(f"[{file_request_id}] Failed to process uploaded image: {e}")
                result["error"] = f"Failed to process uploaded image: {e}"
                results.append(result)
                continue

        elif idx < len(parsed_image_paths):
            path = parsed_image_paths[idx]
            try:
                logger.debug(f"[{file_request_id}] Processing image path: {path}")
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
                logger.error(f"[{file_request_id}] HTTP error processing image: {he.detail}")
                result["error"] = str(he.detail)
                results.append(result)
                continue
            except Exception as e:
                logger.error(f"[{file_request_id}] Failed to process image path: {e}")
                result["error"] = f"Failed to process image path: {e}"
                results.append(result)
                continue

        if not image_processed:
            logger.debug(f"[{file_request_id}] No image available for this JSON file")
            result["image_source"] = "none"

        try:
            logger.debug(f"[{file_request_id}] Calling Gemini API")
            resp = client.models.generate_content(model=GEMINI_MODEL, contents=parts)
            cleaned_text = strip_code_fences(resp.text or "")
            result["result"] = cleaned_text or ""

            file_elapsed = time.time() - file_start_time
            logger.info(f"[{file_request_id}] File processed successfully in {file_elapsed:.2f}s")
        except Exception as e:
            logger.error(f"[{file_request_id}] Gemini API error: {e}")
            result["error"] = f"Gemini API error: {e}"

        results.append(result)

    total_elapsed = time.time() - start_time
    logger.info(f"[{request_id}] Multi-resolve completed in {total_elapsed:.2f}s")

    return results

@app.post("/v1/tags/resolve/upload", response_model=SolveResponse)
async def solve_with_upload(
    json_file: UploadFile = File(...),
    image_file: Optional[UploadFile] = File(None),
    api_key: APIKey = Depends(check_permission("tags:resolve"))
):
    """
    Analyze a DOMX JSON document with direct image upload.
    """
    start_time = time.time()
    request_id = f"resolve_upload_{int(time.time() * 1000)}"

    logger.info(f"[{request_id}] Tags resolve with upload from {api_key.name}")
    logger.info(f"[{request_id}] JSON: {json_file.filename}, Image: {image_file.filename if image_file else 'None'}")

    try:
        raw_json_bytes = await json_file.read()
        json_text = raw_json_bytes.decode("utf-8")
        _ = json.loads(json_text)
        logger.debug(f"[{request_id}] Successfully parsed JSON")
    except Exception as e:
        logger.error(f"[{request_id}] Invalid JSON: {e}")
        raise HTTPException(400, f"Invalid JSON file: {e}")

    parts = [
        SYSTEM_PROMPT,
        f"Here is the JSON document to use:\n```json\n{json_text}\n```",
    ]

    if image_file:
        try:
            logger.debug(f"[{request_id}] Processing uploaded image: {image_file.filename}")
            image_bytes = await image_file.read()
            img = Image.open(BytesIO(image_bytes))

            if img.format not in ['PNG', 'JPEG', 'JPG', 'GIF', 'BMP', 'WEBP']:
                logger.error(f"[{request_id}] Unsupported image format: {img.format}")
                raise HTTPException(400, f"Unsupported image format: {img.format}")

            parts.append(img)
            logger.info(f"[{request_id}] Image processed successfully: {img.format} {img.size}")

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[{request_id}] Failed to process uploaded image: {e}")
            raise HTTPException(400, f"Failed to process uploaded image: {e}")

    try:
        logger.info(f"[{request_id}] Calling Gemini API")
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=parts)
        cleaned_text = strip_code_fences(resp.text or "")

        elapsed = time.time() - start_time
        logger.info(f"[{request_id}] Request completed successfully in {elapsed:.2f}s")

        return SolveResponse(text=cleaned_text or "")
    except Exception as e:
        logger.error(f"[{request_id}] Gemini API error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(500, f"Gemini API error: {e}")

@app.post("/v1/generate-image/file")
async def generate_image_file(
    prompt: str = Form(...),
    input_image: Optional[UploadFile] = File(None),
    api_key: APIKey = Depends(check_permission("image:generate"))
):
    """
    Generate or edit images and return as downloadable file.
    """
    start_time = time.time()
    request_id = f"generate_image_{int(time.time() * 1000)}"

    logger.info(f"[{request_id}] Image generation request from {api_key.name}")
    logger.info(f"[{request_id}] Prompt length: {len(prompt)} chars")

    if api_key.environment == "testing":
        logger.warning(f"[{request_id}] Image generation blocked for testing environment")
        raise HTTPException(
            status_code=403,
            detail="Image generation not available in testing environment"
        )

    try:
        parts = [prompt]

        if input_image and input_image.filename:
            logger.info(f"[{request_id}] Processing input image: {input_image.filename}")

            try:
                image_bytes = await input_image.read()

                if not image_bytes:
                    logger.error(f"[{request_id}] Uploaded file is empty")
                    raise HTTPException(400, "Uploaded file is empty")

                img = Image.open(BytesIO(image_bytes))

                if img.format not in ['PNG', 'JPEG', 'JPG', 'GIF', 'BMP', 'WEBP']:
                    logger.error(f"[{request_id}] Unsupported format: {img.format}")
                    raise HTTPException(400, f"Unsupported image format: {img.format}")

                parts.append(img)
                logger.info(f"[{request_id}] Input image processed: {img.format} {img.size}")

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"[{request_id}] Failed to process input image: {e}")
                raise HTTPException(400, f"Failed to process input image: {str(e)}")

        logger.info(f"[{request_id}] Calling Gemini API for image generation")
        response = client.models.generate_content(
            model=GEMINI_MODEL_IMAGE,
            contents=parts
        )

        generated_image_data = None

        if not response.candidates:
            logger.error(f"[{request_id}] No candidates in generation response")
            raise HTTPException(500, "No candidates in generation response")

        for part in response.candidates[0].content.parts:
            if hasattr(part, 'inline_data') and part.inline_data is not None:
                generated_image_data = part.inline_data.data
                break

        if generated_image_data is None:
            logger.error(f"[{request_id}] No image was generated")
            raise HTTPException(500, "No image was generated in the response")

        elapsed = time.time() - start_time
        logger.info(f"[{request_id}] Image generated successfully in {elapsed:.2f}s")

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
        logger.error(f"[{request_id}] Image generation failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(500, f"Image generation failed: {str(e)}")

@app.post("/v1/image/enhance-prompt")
async def enhance_localization_prompt(
    suggestion: str = Form(...),
    target_locale: str = Form(...),
    website_context: str = Form(...),
    api_key: APIKey = Depends(check_permission("image:analyze"))
):
    """
    Enhance a basic localization suggestion into a detailed image generation prompt.
    """
    start_time = time.time()
    request_id = f"enhance_prompt_{int(time.time() * 1000)}"

    logger.info(f"[{request_id}] Prompt enhancement request from {api_key.name}")
    logger.info(f"[{request_id}] Target locale: {target_locale}, Context: {website_context}")

    enhancement_request = f"""{PROMPT_ENHANCEMENT_GUIDE}

**Input Suggestion:**
{suggestion}

**Target Locale:** {target_locale}
**Website Context:** {website_context}

Transform this suggestion into a detailed, actionable image generation prompt following the enhancement framework."""

    try:
        logger.info(f"[{request_id}] Calling Gemini API for prompt enhancement")
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[enhancement_request]
        )

        enhanced_prompt = response.text or ""

        elapsed = time.time() - start_time
        logger.info(f"[{request_id}] Prompt enhanced successfully in {elapsed:.2f}s")

        return {
            "original_suggestion": suggestion,
            "enhanced_prompt": enhanced_prompt,
            "target_locale": target_locale,
            "website_context": website_context
        }

    except Exception as e:
        logger.error(f"[{request_id}] Gemini API error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(500, f"Gemini API error: {str(e)}")

@app.post("/v1/image/full-localization-pipeline")
async def full_localization_pipeline(
    target_locale: str = Form(...),
    website_context: str = Form(...),
    original_image: Optional[UploadFile] = File(None),
    original_image_path: Optional[str] = Form(None),
    custom_generation_prompt: Optional[str] = Form(None),
    auto_generate: bool = Form(True),
    api_key: APIKey = Depends(check_permission("image:generate"))
):
    """
    Complete localization pipeline: Analyze → Suggest → Generate new image.
    """
    start_time = time.time()
    request_id = f"localization_pipeline_{int(time.time() * 1000)}"

    # Sanitize inputs to prevent header injection issues
    target_locale = target_locale.strip()
    website_context = website_context.strip()

    logger.info(f"[{request_id}] Full localization pipeline from {api_key.name}")
    logger.info(f"[{request_id}] Target: {target_locale}, Context: {website_context}, Auto-generate: {auto_generate}")

    if not original_image and not original_image_path:
        logger.error(f"[{request_id}] No image source provided")
        raise HTTPException(400, "Either 'original_image' file upload or 'original_image_path' must be provided")

    img_to_analyze = None

    # Process image input
    if original_image and original_image.filename:
        try:
            logger.info(f"[{request_id}] Processing uploaded image: {original_image.filename}")
            image_bytes = await original_image.read()
            if not image_bytes:
                raise HTTPException(400, "Uploaded file is empty")

            await original_image.seek(0)
            img_to_analyze = Image.open(BytesIO(image_bytes))

            if img_to_analyze.format not in ['PNG', 'JPEG', 'JPG', 'GIF', 'BMP', 'WEBP']:
                logger.error(f"[{request_id}] Unsupported format: {img_to_analyze.format}")
                raise HTTPException(400, f"Unsupported image format: {img_to_analyze.format}")

            logger.info(f"[{request_id}] Image loaded: {img_to_analyze.format} {img_to_analyze.size}")

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[{request_id}] Failed to process uploaded image: {e}")
            raise HTTPException(400, f"Failed to process uploaded image: {str(e)}")

    elif original_image_path:
        try:
            logger.info(f"[{request_id}] Loading image from path: {original_image_path}")
            if original_image_path.startswith('s3://') or (original_image_path.startswith('https://') and ('s3.' in original_image_path or 's3-' in original_image_path)):
                img_to_analyze = download_s3_image(original_image_path)
            else:
                if not os.path.exists(original_image_path):
                    raise HTTPException(400, f"Local image path not found: {original_image_path}")
                img_to_analyze = Image.open(original_image_path)
            logger.info(f"[{request_id}] Image loaded successfully")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[{request_id}] Failed to process image: {e}")
            raise HTTPException(400, f"Failed to process image: {str(e)}")

    # Calculate the aspect ratio for the image generation step
    if img_to_analyze:
        width, height = img_to_analyze.size
        actual_ratio = width / height
        standard_ratio = get_standard_aspect_ratio(width, height)

    # Step 1: Analyze the image
    logger.info(f"[{request_id}] Step 1: Analyzing image for localization")
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
        logger.info(f"[{request_id}] Analysis complete. Score: {analysis_result.get('overallSuitabilityScore')}")

    except json.JSONDecodeError:
        logger.error(f"[{request_id}] Invalid JSON from analysis")
        raise HTTPException(500, f"Model returned invalid JSON for analysis: {cleaned_analysis}")
    except Exception as e:
        logger.error(f"[{request_id}] Analysis failed: {e}")
        raise HTTPException(500, f"Gemini API error during analysis: {str(e)}")

    # Step 2: Generate suggestions
    logger.info(f"[{request_id}] Step 2: Generating localization suggestions")
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
        logger.info(f"[{request_id}] Suggestions generated successfully")
    except Exception as e:
        logger.error(f"[{request_id}] Suggestions generation failed: {e}")
        raise HTTPException(500, f"Gemini API error during suggestions generation: {str(e)}")

    response_data = {
        "analysis": analysis_result,
        "suggestions": suggestions_text,
        "target_locale": target_locale,
        "website_context": website_context
    }

    # Step 3: Generate new image if requested
    if auto_generate:
        logger.info(f"[{request_id}] Step 3: Generating localized image")

        if custom_generation_prompt and custom_generation_prompt != "string":
            generation_prompt = custom_generation_prompt
            logger.debug(f"[{request_id}] Using custom generation prompt")
        else:
            logger.debug(f"[{request_id}] Creating enhanced generation prompt")
            base_suggestion = f"Use this image as a style reference, adapting it for the target market: {suggestions_text}"

            enhancement_request = f"""{PROMPT_ENHANCEMENT_GUIDE}

            **Input Suggestion:**
            {base_suggestion}

            **Target Locale:** {target_locale}
            **Website Context:** {website_context}

            Based on these localization recommendations:
            {suggestions_text}

            Key improvements needed:"""

            if analysis_result.get("problematicElements"):
                for element in analysis_result["problematicElements"][:5]:
                    if element.get("suggestedChange"):
                        enhancement_request += f"\n- {element['suggestedChange']}"

            enhancement_request += "\n\nTransform this into a detailed, actionable image generation prompt."

            try:
                enhance_response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[enhancement_request]
                )
                generation_prompt = enhance_response.text or ""
                logger.debug(f"[{request_id}] Enhanced prompt created")
            except Exception as e:
                logger.warning(f"[{request_id}] Prompt enhancement failed, using fallback: {e}")
                generation_prompt = (
                    f"Generate a culturally appropriate image for {target_locale} market.\n"
                    f"Website context: {website_context}\n\n"
                    f"Based on these localization recommendations:\n"
                    f"{suggestions_text}\n\n"
                    f"Create a professional, high-quality image that addresses these suggestions."
                )

        parts = [generation_prompt]
        parts.insert(0, "Use this image as a style reference, adapting it for the target market:")
        parts.append(img_to_analyze)
        
        try:
            logger.info(f"[{request_id}] Calling Gemini for image generation")
            gen_response = client.models.generate_content(
                model=GEMINI_MODEL_IMAGE,
                contents=parts,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(
                        aspect_ratio=standard_ratio
                    )
                )
            )

            generated_image_data = None

            if gen_response.candidates:
                for part in gen_response.candidates[0].content.parts:
                    if hasattr(part, 'inline_data') and part.inline_data is not None:
                        generated_image_data = part.inline_data.data
                        break

            if generated_image_data:
                # Sanitize filename for safe use
                safe_locale = re.sub(r'[^\w\s-]', '', target_locale.lower())
                safe_locale = re.sub(r'[-\s]+', '_', safe_locale)
                filename = f"localized_{safe_locale}_generated.png"

                elapsed = time.time() - start_time
                logger.info(f"[{request_id}] Pipeline completed successfully in {elapsed:.2f}s")

                # Sanitize header values - replace spaces and special characters
                safe_target_locale = re.sub(r'[^\w\-.]', '_', target_locale)
                safe_website_context = re.sub(r'[^\w\-.]', '_', website_context[:50])  # Limit length

                return StreamingResponse(
                    BytesIO(generated_image_data),
                    media_type="image/png",
                    headers={
                        "Content-Disposition": f"attachment; filename={filename}",
                        "Content-Type": "image/png",
                        "X-Analysis-Score": str(analysis_result.get("overallSuitabilityScore", "N/A")),
                        "X-Target-Locale": safe_target_locale,
                        "X-Website-Context": safe_website_context
                    }
                )
            else:
                logger.warning(f"[{request_id}] No image was generated")
                response_data["generated_image_available"] = False
                response_data["generation_error"] = "No image was generated"

        except Exception as e:
            logger.error(f"[{request_id}] Image generation failed: {e}")
            response_data["generated_image_available"] = False
            response_data["generation_error"] = str(e)

    elapsed = time.time() - start_time
    logger.info(f"[{request_id}] Pipeline completed in {elapsed:.2f}s (no image generation)")

    return response_data

# Log application shutdown
@app.on_event("shutdown")
def shutdown_event():
    logger.info("=" * 80)
    logger.info("APPLICATION SHUTTING DOWN")
    logger.info("=" * 80)