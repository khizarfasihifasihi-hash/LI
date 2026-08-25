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
    
    "Hiring me", "New job announcement", "Work anniversary", "Achievement/milestone", 
    "Promotion", "Hiring announcement", "Company layoff response",
    "Thought leadership", "Lessons learned / failure story", "Industry trends analysis",
    "Tips/advice", "Book/article review", "Myth busting",
    "Certification completed", "Graduation", "Course recommendation", "Personal transformation",
    "Conference/event recap", "Team appreciation", "Networking/gratitude", 
    "Event invitation", "Webinar announcement",
    "Product launch", "Startup journey update", "Funding round announcement", 
    "Case study/client success", "Behind-the-scenes look", "about news update post on media"
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
    "Quora", "Discord", "Whatsapp", "Telegram", "Fiverr gigs"
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

# Default video canvas per platform - vertical for short-form/story
# platforms, landscape for Youtube, square as a safe fallback elsewhere.
PLATFORM_VIDEO_SIZE = {
    "Linkedin": (1280, 720),
    "Facebook": (1280, 720),
    "Instagram": (1080, 1920),
    "Youtube": (1920, 1080),
    "X (Twitter)": (1280, 720),
    "Threads": (1080, 1920),
    "Tiktok": (1080, 1920),
    "Snapchat": (1080, 1920),
    "Pinterest": (1080, 1920),
    "Reddit": (1280, 720),
    "Quora": (1280, 720),
    "Discord": (1280, 720),
    "Whatsapp": (1080, 1920),
    "Telegram": (1080, 1920),
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

# NEW FEATURE: generate multiple variants at once so the user can pick the
# best one instead of re-rolling one-at-a-time.
num_variants = st.slider(
    "🧪 Number of variants to generate",
    min_value=1,
    max_value=3,
    value=1,
    help="Generate several different takes on the same brief in one go, "
         "shown side-by-side in tabs, and pick your favorite.",
)

img_toggle_col, vid_toggle_col = st.columns(2)
with img_toggle_col:
    generate_image_toggle = st.checkbox(
        "🎨 Also generate a matching image for this post",
        value=False,
        help="Creates an AI image sized for the selected platform, based on the "
             "post's topic, tone, AND the actual generated post text — using "
             "whichever engine is selected in the sidebar. When generating "
             "multiple variants, the image is created for whichever variant "
             "you select as the keeper.",
    )
with vid_toggle_col:
    generate_video_toggle = st.checkbox(
        "🎬 Also generate a short video (image slideshow) for this post",
        value=False,
        help="Storyboards the post into a few scenes, generates an AI image "
             "per scene with the same image engine, and stitches them into "
             "a short Ken-Burns-style slideshow video sized for the "
             "selected platform. Requires the 'moviepy' package "
             "(pip install moviepy) and an ffmpeg binary on this machine.",
    )

if generate_video_toggle:
    vid_scene_col, vid_dur_col = st.columns(2)
    with vid_scene_col:
        video_num_scenes = st.slider("Number of video scenes", 2, 5, 3)
    with vid_dur_col:
        video_seconds_per_scene = st.slider("Seconds per scene", 2, 6, 3)
else:
    video_num_scenes = 3
    video_seconds_per_scene = 3

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

{variant_instruction}

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


def _make_image_for_entry(history_entry: dict, website: str, topic: str, category: str, tone: str, post_text: str,
                           image_engine: str, mistral_api_key: str, mistral_image_model: str) -> None:
    """Generate and attach a matching image to a history entry in-place."""
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


# ---------------------------------------------------------------------------
# Video generation (image-slideshow) helpers
# ---------------------------------------------------------------------------
import re


def split_into_scenes(topic: str, post_text: str, n: int) -> list:
    """Break the topic + generated post copy into n rough scene
    descriptions, so each scene image is grounded in a different beat of
    the story instead of every scene illustrating the same single idea."""
    combined = f"{topic.strip()}. {post_text.strip()}".strip(". ")
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", combined) if s.strip()]
    if not sentences:
        sentences = [topic.strip() or post_text.strip() or "the post topic"]

    if len(sentences) <= n:
        # Pad by repeating/splitting the topic so we always return n scenes.
        scenes = sentences[:]
        while len(scenes) < n:
            scenes.append(sentences[len(scenes) % len(sentences)])
        return scenes[:n]

    # Distribute sentences across n roughly-equal groups, in order.
    chunk_size = max(1, len(sentences) // n)
    scenes = []
    for i in range(n):
        start = i * chunk_size
        end = (start + chunk_size) if i < n - 1 else len(sentences)
        group = sentences[start:end] or [sentences[-1]]
        scenes.append(" ".join(group))
    return scenes


def generate_video_bytes(scene_prompts: list, width: int, height: int, image_engine: str,
                          mistral_key: str, mistral_model: str, seconds_per_scene: int = 3,
                          fps: int = 24, status_cb=None) -> bytes:
    """Generate one AI image per scene prompt (via whichever image engine is
    selected), then stitch them into an MP4 slideshow with a gentle
    Ken-Burns zoom and a cross-fade between scenes. Returns raw MP4 bytes."""
    try:
        from moviepy.editor import ImageClip, concatenate_videoclips, vfx
    except ImportError as exc:
        raise RuntimeError(
            "Video generation needs the 'moviepy' package and an ffmpeg "
            "binary installed on this machine. Install with: "
            "pip install moviepy"
        ) from exc

    import io
    import tempfile

    import numpy as np
    from PIL import Image

    clips = []
    for i, scene_prompt in enumerate(scene_prompts):
        if status_cb:
            status_cb(f"Generating scene {i + 1}/{len(scene_prompts)}...")
        img_bytes = generate_image(scene_prompt, width, height, image_engine, mistral_key, mistral_model)
        frame = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((width, height)))

        clip = ImageClip(frame).set_duration(seconds_per_scene)
        # Subtle Ken-Burns zoom so a static image doesn't feel like a slide.
        clip = clip.fx(vfx.resize, lambda t: 1.0 + 0.04 * (t / seconds_per_scene))
        clip = clip.set_position(("center", "center")).fx(vfx.crossfadein, min(0.6, seconds_per_scene / 3))
        clips.append(clip)

    if status_cb:
        status_cb("Stitching scenes into the final video...")

    final = concatenate_videoclips(clips, method="compose", padding=-min(0.6, seconds_per_scene / 3))
    final = final.resize((width, height))

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        final.write_videofile(
            tmp_path, fps=fps, codec="libx264", audio=False,
            verbose=False, logger=None,
        )
        with open(tmp_path, "rb") as f:
            video_bytes = f.read()
    finally:
        for clip in clips:
            clip.close()
        final.close()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return video_bytes


def _make_video_for_entry(history_entry: dict, website: str, topic: str, category: str, tone: str, post_text: str,
                           image_engine: str, mistral_api_key: str, mistral_image_model: str,
                           num_scenes: int, seconds_per_scene: int) -> None:
    """Storyboard the post into scenes, generate a scene image for each,
    stitch them into a slideshow video, and attach it to a history entry."""
    vid_width, vid_height = PLATFORM_VIDEO_SIZE.get(website, (1080, 1920))
    aspect_label = _aspect_label(vid_width, vid_height)
    scenes = split_into_scenes(topic, post_text, num_scenes)
    scene_prompts = [
        build_image_prompt(topic, category, tone, scene, aspect_label) for scene in scenes
    ]
    state_key = f"post_video_{len(st.session_state.history)}"
    status_box = st.empty()
    try:
        video_bytes = generate_video_bytes(
            scene_prompts, vid_width, vid_height, image_engine,
            mistral_api_key, mistral_image_model, seconds_per_scene=seconds_per_scene,
            status_cb=lambda msg: status_box.info(f"🎬 {msg}"),
        )
        st.session_state[state_key] = video_bytes
        history_entry["video_key"] = state_key
        history_entry["video_scenes"] = scenes
        history_entry["image_engine_for_video"] = image_engine
    except Exception as e:
        st.error(f"Video generation failed: {e}")
    finally:
        status_box.empty()


# ---------------------------------------------------------------------------
# Generation Logic Engine
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "pending_variants" not in st.session_state:
    st.session_state.pending_variants = []
if "pending_config" not in st.session_state:
    st.session_state.pending_config = {}

generate = st.button("🚀 Generate post", type="primary", use_container_width=True)

# Slight variety hints so multiple variants don't come back near-identical
# even at the same temperature.
VARIANT_ANGLES = [
    "Take a direct, get-to-the-point angle.",
    "Lead with a hook, anecdote, or surprising observation before the main point.",
    "Frame it around a question or a lesson learned.",
]

if generate:
    if not api_key:
        st.error("Please enter your Groq API key in the sidebar (or set GROQ_API_KEY).")
    elif not topic.strip():
        st.error("Please describe what the post should be about.")
    elif (generate_image_toggle or generate_video_toggle) and image_engine.startswith("Mistral") and not mistral_api_key:
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
            chain = build_chain(api_key, model, temperature)
            variants = []
            with st.spinner(
                f"Writing {num_variants} variant{'s' if num_variants > 1 else ''}..."
            ):
                for i in range(num_variants):
                    variant_instruction = (
                        VARIANT_ANGLES[i % len(VARIANT_ANGLES)] if num_variants > 1 else ""
                    )
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
                            "variant_instruction": variant_instruction,
                            "example_block": example_block,
                        }
                    )
                    variants.append(post_text)

            # Stash for the picker UI below so a rerun (e.g. from clicking
            # "Use this variant") doesn't lose the generated options.
            st.session_state.pending_variants = variants
            st.session_state.pending_config = {
                "website": website,
                "topic": topic,
                "category": category,
                "tone": tone,
                "generate_image_toggle": generate_image_toggle,
                "generate_video_toggle": generate_video_toggle,
                "video_num_scenes": video_num_scenes,
                "video_seconds_per_scene": video_seconds_per_scene,
                "image_engine": image_engine,
                "mistral_api_key": mistral_api_key,
                "mistral_image_model": mistral_image_model,
            }

        except Exception as e:
            st.error(f"An error occurred during LLM text generation: {e}")

# Show whichever variants are pending selection (persists across reruns).
if st.session_state.pending_variants:
    cfg = st.session_state.pending_config
    variants = st.session_state.pending_variants

    st.subheader("✨ Generated Output")
    if len(variants) == 1:
        st.text_area("Copy your post text:", value=variants[0], height=350)
        chosen_text = variants[0]
        use_clicked = True
    else:
        tabs = st.tabs([f"Variant {i+1}" for i in range(len(variants))])
        chosen_text = None
        use_clicked = False
        for i, (tab, text) in enumerate(zip(tabs, variants)):
            with tab:
                st.text_area("Copy this variant:", value=text, height=300, key=f"variant_text_{i}")
                if st.button(f"✅ Use Variant {i+1}", key=f"use_variant_{i}"):
                    chosen_text = text
                    use_clicked = True

    if use_clicked and chosen_text:
        history_entry = {"platform": cfg["website"], "content": chosen_text, "image_key": None, "video_key": None}

        if cfg["generate_image_toggle"]:
            _make_image_for_entry(
                history_entry, cfg["website"], cfg["topic"], cfg["category"], cfg["tone"],
                chosen_text, cfg["image_engine"], cfg["mistral_api_key"], cfg["mistral_image_model"],
            )

        if cfg.get("generate_video_toggle"):
            _make_video_for_entry(
                history_entry, cfg["website"], cfg["topic"], cfg["category"], cfg["tone"],
                chosen_text, cfg["image_engine"], cfg["mistral_api_key"], cfg["mistral_image_model"],
                cfg["video_num_scenes"], cfg["video_seconds_per_scene"],
            )

        st.session_state.history.append(history_entry)
        st.session_state.pending_variants = []
        st.session_state.pending_config = {}
        st.rerun()

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
            video_key = historical_post.get("video_key")
            if video_key and video_key in st.session_state:
                st.video(st.session_state[video_key])
                st.download_button(
                    "⬇️ Download video",
                    data=st.session_state[video_key],
                    file_name=f"{video_key}.mp4",
                    mime="video/mp4",
                    key=f"dl_history_{video_key}",
                )
