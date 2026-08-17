import os
import json
import random
from urllib.parse import quote

import requests
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LIF.ai",
    page_icon="🚀",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Expanded Option Lists
# ---------------------------------------------------------------------------
CATEGORIES = [
    "New job announcement", "Work anniversary", "Achievement / milestone", 
    "Promotion", "Hiring announcement", "Company layoff response",
    "Thought leadership", "Lessons learned / failure story", "Industry trends analysis",
    "Tips / advice", "Book / article review", "Myth busting",
    "Certification completed", "Graduation", "Course recommendation", "Personal transformation",
    "Conference / event recap", "Team appreciation", "Networking / gratitude", 
    "Event invitation", "Webinar announcement",
    "Product launch", "Startup journey update", "Funding round announcement", 
    "Case study / client success", "Behind-the-scenes look", "about news update post on media"
]

TONES = [
    "Professional", "Casual & friendly", "Inspirational", "Storytelling", "Bold & confident",
    "Humorous & witty", "Empathetic & vulnerable", "Educational & authoritative", 
    "Contrarian / provocative", "Urgent & hype-building"
]

LANGUAGES = [
    "English", "Urdu", "Roman Urdu (Urdu written in English letters)", "Arabic", 
    "Spanish", "French", "Hindi", "German", "Portuguese", "Turkish", 
    "Indonesia", "Russian", "Persian",
]

WEBSITE = [
    "Linkedin", "Facebook", "Instagram", "Youtube", "X (Twitter)", 
    "Threads", "Tiktok", "Snapchat", "Pinterest", "Reddit", 
    "Quora", "Discord", "Whatsapp", "Telegram"
]

# Sensible default image canvas per platform - a square Instagram crop and a
# vertical TikTok crop need very different dimensions, so pick a size that
# matches how each platform actually displays images.
PLATFORM_IMAGE_SIZE = {
    "Linkedin": (1200, 627),
    "Facebook": (1200, 630),
    "Instagram": (1080, 1080),
    "Youtube": (1280, 720),
    "X (Twitter)": (1200, 675),
    "Threads": (1080, 1080),
    "Tiktok": (1080, 1920),
    "Snapchat": (1080, 1920),
    "Pinterest": (1000, 1500),
    "Reddit": (1200, 675),
    "Quora": (1200, 675),
    "Discord": (1200, 675),
    "Whatsapp": (1080, 1080),
    "Telegram": (1080, 1080),
}

# Rough aspect-ratio label per platform, used only to *describe* the desired
# framing to Mistral's image tool (which - unlike Pollinations - doesn't take
# explicit width/height parameters, only prompt text).
def _aspect_label(width: int, height: int) -> str:
    ratio = width / height
    if abs(ratio - 1) < 0.05:
        return "square (1:1)"
    if ratio > 1:
        return "landscape / horizontal, wide format"
    return "portrait / vertical, tall format"


# ---------------------------------------------------------------------------
# Sidebar Settings & Website Directory List
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input(
        "Groq API Key",
        value=os.environ.get("GROQ_API_KEY", ""),
        type="password",
        help="Get a free key at https://groq.com. "
             "You can also set it as the GROQ_API_KEY environment variable "
             "instead of pasting it here.",
    )
    model = st.selectbox(
        "Model",
        [
            "openai/gpt-oss-120b",  # High intelligence replacement for 70B
            
            "openai/gpt-oss-20b",   # Ultra-fast replacement for 8B
            "qwen/qwen3.6-27b",     # Highly balanced reasoning model
        ],
        index=0,
        )
    
    temperature = st.slider("Creativity (temperature)", 0.0, 1.5, 0.8, 0.1)

    st.markdown("---")
    st.subheader("🖼️ Image generation engine")
    image_engine = st.radio(
        "Which model should draw the image?",
        ["Pollinations.ai (free, no key)", "Mistral (Agents API + Flux Ultra)"],
        index=0,
        help="Pollinations is a free keyless image API. The Mistral option "
             "routes image creation through Mistral's own Agents API "
             "(image_generation tool, Black Forest Labs FLUX1.1 [pro] Ultra) "
             "using the SAME post-writing context, so text and image come "
             "from one connected pipeline.",
    )
    mistral_api_key = ""
    if image_engine.startswith("Mistral"):
        mistral_api_key = st.text_input(
            "Mistral API Key",
            value=os.environ.get("MISTRAL_API_KEY", ""),
            type="password",
            help="Get a key at https://console.mistral.ai. Required only "
                 "when the Mistral image engine is selected.",
        )
        mistral_image_model = st.selectbox(
            "Mistral agent model",
            ["mistral-medium-latest", "mistral-large-latest", "mistral-small-latest"],
            index=0,
            help="The LLM that drives the image-generation agent.",
        )
    else:
        mistral_image_model = "mistral-medium-latest"

    st.caption(
        "🖼️ Pollinations.ai needs no key. Mistral's image tool needs a "
        "Mistral API key and bills through your Mistral account."
    )
    

img_col, title_col = st.columns([1, 4])
with img_col:
    try:
        st.image("ChatGPT.png", width=120)
    except Exception:
        st.text("📝 LIF.ai Logo")

with title_col:
    st.title("LIF.ai")
    st.caption("Powered by LangChain + Groq — turn a rough idea into a polished marketing post in seconds.")

st.markdown("---")

# Input Configuration Selectors (Fixed Column matching 3 vs 3)
col1, col2, col3 = st.columns(3)
with col1:
    category = st.selectbox("Post type", CATEGORIES)
with col2:
    tone = st.selectbox("Tone", TONES)
with col3:
    website = st.selectbox("Web platform target", WEBSITE, index=0) # Fixed typo 'seletbox'

topic = st.text_area(
    "Topic — what's the post about?",
    placeholder="e.g. I just shipped a new feature that reduced load times by 40%...",
    height=120,
)

# Output Style Controls (Fixed duplicate col3 variable naming conflicts)
col_len, col_lang, col_opts = st.columns(3)
with col_len:
    length = st.selectbox("Length", ["Short", "Medium", "Long"], index=1)
with col_lang:
    language = st.selectbox("Language", LANGUAGES, index=0)
with col_opts:
    st.write("**Formatting Toggles**")
    use_emojis = st.checkbox("Emojis", value=True)
    use_hashtags = st.checkbox("Include hashtags", value=True)

# Optional few-shot dataset switch
use_examples = st.checkbox(
    "Use style examples from linkedin_post_dataset.json (if present in this folder)",
    value=False,
)

generate_image_toggle = st.checkbox(
    "🎨 Also generate a matching image for this post",
    value=False,
    help="Creates an AI image sized for the selected platform, based on the "
         "post's topic, tone, AND the actual generated post text — using "
         "whichever engine is selected in the sidebar.",
)

def load_examples(category_name: str, n: int = 2):
    path = os.path.join(os.path.dirname(__file__), "linkedin_post_dataset.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    
    # Fixed matching bug: checking flat string sub-matches safely
    category_slug = category_name.lower()
    matches = [d for d in data if category_slug in d.get("category", "").lower()]
    if not matches:
        matches = data
    return matches[:n]

LENGTH_MAP = {
    "Short": "under 80 words",
    "Medium": "120-180 words",
    "Long": "220-360 words",
}

# Fixed: System and User prompt configurations are streamlined to reference selected platform variable 
SYSTEM_PROMPT = (
    "You are an expert copywriter and social media ghostwriter. You write authentic, engaging, "
    "human-sounding content tailored perfectly for the requested platform. Avoid generic corporate "
    "clichés or extreme clickbait patterns. Return ONLY the finished post text, with no preamble, "
    "no conversational transition chat, and no surrounding quotation marks."
)

USER_PROMPT_TEMPLATE = """Write an optimization-driven post specifically designed for {website}.
Post Category Type: {category}
Desired Tone: {tone}. 
Target Length: {length_desc}. 
Language Constraint: Write the ENTIRE text exclusively in {language}. 

Formatting Rules:
- Use clean layout patterns, short paragraphs, and distinct line breaks to maximize readability.
- {emoji_instruction} 
- {hashtag_instruction} 

Core Content Context Topic:
{topic} 

{example_block}"""

def build_chain(api_key: str, model: str, temperature: float):
    llm = ChatGroq(api_key=api_key, model=model, temperature=temperature)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("user", USER_PROMPT_TEMPLATE),
        ]
    )
    return prompt | llm | StrOutputParser()


def build_image_prompt(topic: str, category: str, tone: str, post_text: str = "", aspect_label: str = "") -> str:
    """Turn the post's topic/category/tone - AND the actual generated post
    copy - into a text-to-image prompt. This is what "connects" the two
    generation steps: the image is grounded in what the post actually says,
    not just the raw topic field, so the visual and the caption stay on the
    same message."""
    base = topic.strip() or category
    prompt = (
        f"A polished, professional social media graphic illustrating: {base}. "
        f"Visual style: clean, modern, {tone.lower()} mood, well suited for a "
        f"{category.lower()} post. No embedded text, no watermarks, no logos."
    )
    if post_text.strip():
        # Keep this short - we only want the gist/keywords of the post as
        # visual grounding, not the literal caption rendered as an image.
        gist = " ".join(post_text.strip().split()[:40])
        prompt += f" The image should visually reflect the theme of this caption: {gist}"
    if aspect_label:
        prompt += f" Composition/framing: {aspect_label}."
    return prompt


def generate_image_bytes(prompt: str, width: int, height: int) -> bytes:
    """Generate an image from a text prompt using Pollinations' free,
    keyless image API (https://image.pollinations.ai). Returns raw image
    bytes (or raises on failure/timeout, which the caller should catch)."""
    params = {
        "width": width,
        "height": height,
        "seed": random.randint(1, 999_999_999),
        "nologo": "true",
        "model": "flux",
    }
    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"https://image.pollinations.ai/prompt/{quote(prompt)}?{query}"
    response = requests.get(url, timeout=90)
    response.raise_for_status()
    return response.content


def get_mistral_image_agent(api_key: str, agent_model: str):
    """Create (once per session) a Mistral agent wired with the built-in
    image_generation connector, and cache its id so every image in this
    session reuses the same agent instead of re-creating one each call."""
    try:
        from mistralai import Mistral
    except ImportError:
        # mistralai >=2.x restructured its package as a namespace package;
        # the client class moved under `mistralai.client`.
        from mistralai.client import Mistral

    cache_key = f"_mistral_agent_{agent_model}"
    client = Mistral(api_key=api_key)
    if cache_key not in st.session_state:
        agent = client.beta.agents.create(
            model=agent_model,
            name="LIF.ai Image Generation Agent",
            description="Generates on-brand marketing images for LIF.ai posts.",
            instructions=(
                "Use the image generation tool whenever asked to create an "
                "image. Produce clean, professional marketing/social-media "
                "visuals with no embedded text, watermarks, or logos."
            ),
            tools=[{"type": "image_generation"}],
            completion_args={"temperature": 0.3, "top_p": 0.95},
        )
        st.session_state[cache_key] = agent.id
    return client, st.session_state[cache_key]


def generate_image_bytes_mistral(prompt: str, api_key: str, agent_model: str) -> bytes:
    """Generate an image via Mistral's Agents API (image_generation
    connector, Black Forest Labs FLUX1.1 [pro] Ultra under the hood).
    Mistral's image tool doesn't take explicit width/height - framing is
    steered through the prompt text instead. Returns raw image bytes."""
    if not api_key:
        raise ValueError("Mistral API key is required for this image engine.")

    client, agent_id = get_mistral_image_agent(api_key, agent_model)
    response = client.beta.conversations.start(agent_id=agent_id, inputs=prompt)

    file_id = None
    for entry in response.outputs:
        content = getattr(entry, "content", None)
        if not content:
            continue
        chunks = content if isinstance(content, list) else [content]
        for chunk in chunks:
            if getattr(chunk, "type", None) == "tool_file":
                file_id = chunk.file_id
                break
        if file_id:
            break

    if not file_id:
        raise RuntimeError("Mistral agent did not return a generated image file.")

    return client.files.download(file_id=file_id).read()


def generate_image(prompt: str, width: int, height: int, engine: str, mistral_key: str, mistral_model: str) -> bytes:
    """Single entry point that routes to whichever engine is selected in
    the sidebar, so the rest of the app doesn't need to branch on it."""
    if engine.startswith("Mistral"):
        return generate_image_bytes_mistral(prompt, mistral_key, mistral_model)
    return generate_image_bytes(prompt, width, height)


def render_image_block(prompt: str, width: int, height: int, state_key: str, engine: str, mistral_key: str, mistral_model: str) -> None:
    """Show the generated image (if any) for `state_key`, plus a
    regenerate button that rerolls (fresh random seed for Pollinations,
    fresh agent turn for Mistral)."""
    if state_key in st.session_state:
        st.image(
            st.session_state[state_key],
            caption=f"AI-generated image ({width}x{height}) · {engine.split(' ')[0]}",
            use_container_width=True,
        )
        st.download_button(
            "⬇️ Download image",
            data=st.session_state[state_key],
            file_name=f"{state_key}.jpg",
            mime="image/jpeg",
            key=f"dl_{state_key}",
        )
    if st.button("🔁 Regenerate image", key=f"regen_{state_key}"):
        with st.spinner("Generating a new image..."):
            try:
                st.session_state[state_key] = generate_image(
                    prompt, width, height, engine, mistral_key, mistral_model
                )
                st.rerun()
            except Exception as e:
                st.error(f"Image generation failed: {e}")


# ---------------------------------------------------------------------------
# Generation Logic Engine
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

generate = st.button("🚀 Generate post", type="primary", use_container_width=True)

if generate:
    if not api_key:
        st.error("Please enter your Groq API key in the sidebar (or set GROQ_API_KEY).")
    elif not topic.strip():
        st.error("Please describe what the post should be about.")
    elif generate_image_toggle and image_engine.startswith("Mistral") and not mistral_api_key:
        st.error("Please enter your Mistral API key in the sidebar, or switch to the Pollinations image engine.")
    else:
        emoji_instruction = (
            "Include a few relevant emojis, used sparingly." if use_emojis else "Do not use any emojis."
        )
        hashtag_instruction = (
            "End with 3-5 relevant hashtags." if use_hashtags else "Do not include hashtags."
        )
        
        example_block = ""
        if use_examples:
            examples = load_examples(category)
            if examples:
                joined = "\n\n".join(f"Example post:\n{ex['post']}" for ex in examples)
                example_block = (
                    "Here are a couple of example posts in a similar style for reference "
                    "(do not copy them, just match the tone and structure):\n\n" + joined
                )
        
        try:
            with st.spinner("Writing your post..."):
                chain = build_chain(api_key, model, temperature)
                post_text = chain.invoke(
                    {
                        "website": website,
                        "category": category,
                        "tone": tone,
                        "length_desc": LENGTH_MAP[length],
                        "language": language,
                        "emoji_instruction": emoji_instruction,
                        "hashtag_instruction": hashtag_instruction,
                        "topic": topic,
                        "example_block": example_block
                    }
                )
                
                # Display output window
                st.subheader("✨ Generated Output")
                st.text_area("Copy your post text:", value=post_text, height=350)

                history_entry = {"platform": website, "content": post_text, "image_key": None}

                # Optionally generate a matching image right away - grounded
                # in BOTH the topic and the post text the LLM just wrote, so
                # the two generation steps are genuinely connected.
                if generate_image_toggle:
                    img_width, img_height = PLATFORM_IMAGE_SIZE.get(website, (1080, 1080))
                    aspect_label = _aspect_label(img_width, img_height)
                    image_prompt = build_image_prompt(topic, category, tone, post_text, aspect_label)
                    state_key = f"post_image_{len(st.session_state.history)}"
                    with st.spinner(f"Generating a matching image with {image_engine.split(' ')[0]}..."):
                        try:
                            st.session_state[state_key] = generate_image(
                                image_prompt, img_width, img_height,
                                image_engine, mistral_api_key, mistral_image_model,
                            )
                            history_entry["image_key"] = state_key
                            history_entry["image_prompt"] = image_prompt
                            history_entry["image_engine"] = image_engine
                        except Exception as e:
                            st.error(f"Image generation failed: {e}")

                # Append to active session history state
                st.session_state.history.append(history_entry)

                # Show the freshly generated image (if any) right below the post
                if history_entry["image_key"]:
                    img_width, img_height = PLATFORM_IMAGE_SIZE.get(website, (1080, 1080))
                    st.subheader("🖼️ Generated Image")
                    render_image_block(
                        history_entry["image_prompt"],
                        img_width,
                        img_height,
                        history_entry["image_key"],
                        image_engine,
                        mistral_api_key,
                        mistral_image_model,
                    )
                
        except Exception as e:
            st.error(f"An error occurred during LLM text generation: {e}")

# Render basic log list of generated posts if any exist
if st.session_state.history:
    st.markdown("---")
    st.subheader("📚 Generation History Session Logs")
    for idx, historical_post in enumerate(reversed(st.session_state.history)):
        real_idx = len(st.session_state.history) - idx
        with st.expander(f"Post {real_idx} - Platform: {historical_post['platform']}"):
            st.code(historical_post['content'], language="text")
            image_key = historical_post.get("image_key")
            if image_key and image_key in st.session_state:
                st.image(st.session_state[image_key], use_container_width=True)
                st.download_button(
                    "⬇️ Download image",
                    data=st.session_state[image_key],
                    file_name=f"{image_key}.jpg",
                    mime="image/jpeg",
                    key=f"dl_history_{image_key}",
                )