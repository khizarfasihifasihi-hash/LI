import os
import json

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
# ---------------------------------------------------------------------------
# Expanded Option Lists (upgraded)
# ---------------------------------------------------------------------------


CATEGORIES = [
    'Posting on Website',
    'Hiring me',
    'New job announcement',
    'Work anniversary',
    'Achievement/milestone',
    'Promotion',
    'Hiring announcement',
    'Company layoff response',
    'Thought leadership',
    'Lessons learned / failure story',
    'Industry trends analysis',
    'Tips/advice',
    'Book/article review',
    'Myth busting',
    'Certification completed',
    'Graduation',
    'Course recommendation',
    'Personal transformation',
    'Conference/event recap',
    'Team appreciation',
    'Networking/gratitude',
    'Event invitation',
    'Webinar announcement',
    'Product launch',
    'Startup journey update',
    'Funding round announcement',
    'Case study/client success',
    'Behind-the-scenes look',
    'News/media update post',
    'Product showcase',
    'New arrival announcement',
    'Product restock',
    'Flash sale/discount',
    'Limited-time offer',
    'Product bundle deal',
    'Customer review/testimonial highlight',
    'Unboxing/first look',
    'How-to / product tutorial',
    'Before & after',
    'Seasonal collection drop',
    'Gift guide',
    'Comparison (this vs that)',
    'FAQ / product Q&A',
    'Out of stock / back-order notice',
    'Pre-order announcement',
    'Waitlist announcement',
    'Price drop announcement',
    'Shipping delay notice',
    'Return/refund policy update',
    'Poll/question to audience',
    'This or that',
    'Fun fact / trivia',
    'Motivational quote',
    'Holiday/seasonal greeting',
    'Company milestone',
    'Partnership/collaboration announcement',
    'Recruitment drive',
    'Internship program announcement',
    'Community shoutout',
    'User-generated content repost',
    'Apology/service update',
    'Repost/share with commentary',
    'AMA (Ask Me Anything)',
    'Live event coverage',
    'Year in review',
    'Prediction/forecast post',
    'Controversial/hot take opinion piece',
    'New feature announcement',
    'Bug fix / changelog update',
    'Open source release',
    'Job vacancy detail post',
    'Employee spotlight',
    'Company culture post',
    'Sustainability/CSR update',
    'Charity/donation drive',
    'Survey results share',
    'Interview highlights',
    'Podcast episode promo',
    'Newsletter signup push',
    'Loyalty program announcement',
    'Referral program push',
    'App update / new version release',
    'Customer support tip',
    'Success metrics/KPI share',
    'Meme/relatable content',
    'Client onboarding welcome',
    'Vendor/supplier spotlight',
    'Investor update',
    'Board announcement',
    'Executive appointment',
    'Retirement announcement',
    'Office relocation announcement',
    'New market entry announcement',
    'International expansion update',
    'Award/recognition received',
    'Speaking engagement announcement',
    'Panel discussion recap',
    'Research paper/whitepaper release',
    'Data report release',
    'Trend prediction for the year',
    'Industry award nomination',
    'Customer complaint response',
    'Crisis communication update',
    'Product recall notice',
    'Security incident disclosure',
    'Privacy policy update',
    'Terms of service update',
    'Feature deprecation notice',
    'Beta program invitation',
    'Early access announcement',
    'Referral success story',
    'Affiliate program announcement',
    'Reseller partnership announcement',
    'Franchise opportunity post',
    'Store opening announcement',
    'Store closing announcement',
    'Pop-up event announcement',
    'Trade show booth announcement',
    'Sponsorship announcement',
    'Scholarship announcement',
    'Mentorship program launch',
    'Hackathon announcement',
    'Contest/giveaway announcement',
    'Giveaway winner announcement',
    'Customer anniversary shoutout',
    'Milestone user count celebration',
    'App store rating request',
    'Referral leaderboard update',
    'Subscription plan change',
    'Price increase explanation',
    'New pricing tier announcement',
    'Free trial offer',
    'Upgrade prompt post',
    'Cross-sell/upsell post',
    'Bundle savings highlight',
    'Clearance sale announcement',
    'End of season sale',
    'Back to school promotion',
    'Black Friday/Cyber Monday promo',
    'New Year resolution post',
    'Anniversary sale',
    'Loyalty milestone celebration',
    'Customer FAQ roundup',
    'Common mistake to avoid post',
    'Industry statistic share',
    'Behind the recipe/process reveal',
    'Team building event recap',
    'Company retreat recap',
    'Volunteer day recap',
    'Diversity & inclusion update',
    'Environmental impact report',
    'Product sustainability feature',
    'Packaging redesign announcement',
    'Supply chain update',
    'Local business spotlight',
    'Customer journey story',
    'Founder story',
    'Origin story post',
    'Vision/mission statement post',
    'Company values post',
    'Rebrand announcement',
    'Logo/visual identity update',
    'Website relaunch announcement',
    'App redesign announcement',
    'New office tour',
    'Remote work culture post',
    'Work-from-home tips post',
    'Productivity tips post',
    'Industry glossary/explainer post',
    'Common myths in the industry post',
    'Expert roundup post',
    'Guest post announcement',
    'Collaboration reveal',
    'Influencer partnership announcement',
    'Brand ambassador announcement',
    'Customer spotlight video promo',
    'Testimonial video release',
    'Case study video release',
    'Explainer video release',
    'Live Q&A announcement',
    'Twitter Spaces/audio room announcement',
    'Community event recap',
    'User milestone celebration',
    'API release announcement',
    'Integration partnership announcement',
    'Platform outage notice',
    'Service restoration update',
    'Maintenance window notice',
    'New hire welcome post',
    'Internal promotion celebration',
    'Team expansion announcement',
    'Culture award received',
    'Best place to work recognition',
    'Customer education series post',
    'Weekly roundup post',
    'Monthly recap post',
    'Quarterly results share',
    'Annual report highlight',
    'End of year thank you post',
    'New Year kickoff post',
]


TONES = [
    'Professional',
    'Casual & friendly',
    'Inspirational',
    'Storytelling',
    'Bold & confident',
    'Humorous & witty',
    'Empathetic & vulnerable',
    'Educational & authoritative',
    'Contrarian / provocative',
    'Urgent & hype-building',
    'Playful & quirky',
    'Luxurious & aspirational',
    'Minimal & understated',
    'Warm & conversational',
    'Sarcastic/deadpan',
    'Nostalgic',
    'Data-driven & analytical',
    'Reassuring & trustworthy',
    'Edgy & rebellious',
    'Celebratory',
    'Formal & corporate',
    'Poetic/lyrical',
    'Direct & no-nonsense',
    'Curious & exploratory',
    'Encouraging & supportive',
    'Witty & clever',
    'Grateful & humble',
    'Confident & assertive',
    'Relatable & down-to-earth',
    'Futuristic & visionary',
    'Calm & mindful',
    'Punchy & headline-style',
    'Whimsical & lighthearted',
    'Serious & no-frills',
    'Optimistic & upbeat',
    'Skeptical & analytical',
    'Reflective & introspective',
    'Energetic & enthusiastic',
    'Diplomatic & measured',
    'Blunt & candid',
    'Wholesome & feel-good',
    'Mysterious & intriguing',
    'Dramatic & theatrical',
    'Friendly & approachable',
    'Authoritative & commanding',
    'Curious & question-driven',
    'Journalistic & neutral',
    'Conversational & casual-formal blend',
    'Empowering & motivational',
    'Comforting & gentle',
    'Snarky & irreverent',
    'Elegant & refined',
    'Rustic & down-home',
    'Tech-savvy & cutting-edge',
    'Old-school & traditional',
    'Youthful & trendy',
    'Wise & mentor-like',
    'Cheeky & flirtatious',
    'Somber & respectful',
    'Triumphant & victorious',
    'Humble-brag',
    'Matter-of-fact',
    'Persuasive & sales-forward',
    'Community-driven & inclusive',
    'Exclusive & insider-tone',
    'Urgent & scarcity-driven',
    'Playful & meme-forward',
    'Heartfelt & sincere',
    'Analytical & research-backed',
    'Bold & controversial',
    'Soft-sell & subtle',
]

LANGUAGES = [
    ("Abkhaz", "ab"),
    ("Afar", "aa"),
    ("Afrikaans", "af"),
    ("Akan", "ak"),
    ("Albanian", "sq"),
    ("Amharic", "am"),
    ("Arabic", "ar"),
    ("Aragonese", "an"),
    ("Armenian", "hy"),
    ("Assamese", "as"),
    ("Avaric", "av"),
    ("Avestan", "ae"),
    ("Aymara", "ay"),
    ("Azerbaijani", "az"),
    ("Bambara", "bm"),
    ("Bashkir", "ba"),
    ("Basque", "eu"),
    ("Belarusian", "be"),
    ("Bengali", "bn"),
    ("Bihari", "bh"),
    ("Bislama", "bi"),
    ("Bosnian", "bs"),
    ("Breton", "br"),
    ("Bulgarian", "bg"),
    ("Burmese", "my"),
    ("Catalan", "ca"),
    ("Chamorro", "ch"),
    ("Chechen", "ce"),
    ("Chichewa", "ny"),
    ("Chinese (Simplified)", "zh-Hans"),
    ("Chinese (Traditional)", "zh-Hant"),
    ("Chuvash", "cv"),
    ("Cornish", "kw"),
    ("Corsican", "co"),
    ("Cree", "cr"),
    ("Croatian", "hr"),
    ("Czech", "cs"),
    ("Danish", "da"),
    ("Divehi", "dv"),
    ("Dutch", "nl"),
    ("Dzongkha", "dz"),
    ("English", "en"),
    ("Esperanto", "eo"),
    ("Estonian", "et"),
    ("Ewe", "ee"),
    ("Faroese", "fo"),
    ("Fijian", "fj"),
    ("Finnish", "fi"),
    ("French", "fr"),
    ("Fulah", "ff"),
    ("Galician", "gl"),
    ("Georgian", "ka"),
    ("German", "de"),
    ("Greek", "el"),
    ("Guarani", "gn"),
    ("Gujarati", "gu"),
    ("Haitian Creole", "ht"),
    ("Hausa", "ha"),
    ("Hebrew", "he"),
    ("Herero", "hz"),
    ("Hindi", "hi"),
    ("Hiri Motu", "ho"),
    ("Hungarian", "hu"),
    ("Interlingua", "ia"),
    ("Indonesian", "id"),
    ("Interlingue", "ie"),
    ("Irish", "ga"),
    ("Igbo", "ig"),
    ("Inupiaq", "ik"),
    ("Ido", "io"),
    ("Icelandic", "is"),
    ("Italian", "it"),
    ("Inuktitut", "iu"),
    ("Japanese", "ja"),
    ("Javanese", "jv"),
    ("Kalaallisut", "kl"),
    ("Kannada", "kn"),
    ("Kanuri", "kr"),
    ("Kashmiri", "ks"),
    ("Kazakh", "kk"),
    ("Central Khmer", "km"),
    ("Kikuyu", "ki"),
    ("Kinyarwanda", "rw"),
    ("Kirghiz", "ky"),
    ("Komi", "kv"),
    ("Kongo", "kg"),
    ("Korean", "ko"),
    ("Kurdish", "ku"),
    ("Kuanyama", "kj"),
    ("Latin", "la"),
    ("Luxembourgish", "lb"),
    ("Ganda", "lg"),
    ("Limburgan", "li"),
    ("Lingala", "ln"),
    ("Lao", "lo"),
    ("Lithuanian", "lt"),
    ("Luba-Katanga", "lu"),
    ("Latvian", "lv"),
    ("Manx", "gv"),
    ("Macedonian", "mk"),
    ("Malagasy", "mg"),
    ("Malay", "ms"),
    ("Malayalam", "ml"),
    ("Maltese", "mt"),
    ("Maori", "mi"),
    ("Marathi", "mr"),
    ("Marshallese", "mh"),
    ("Mongolian", "mn"),
    ("Nauru", "na"),
    ("Navajo", "nv"),
    ("North Ndebele", "nd"),
    ("Nepali", "ne"),
    ("Ndonga", "ng"),
    ("Norwegian Bokmål", "nb"),
    ("Norwegian Nynorsk", "nn"),
    ("Norwegian", "no"),
    ("Sichuan Yi", "ii"),
    ("South Ndebele", "nr"),
    ("Occitan", "oc"),
    ("Ojibwa", "oj"),
    ("Church Slavic", "cu"),
    ("Oromo", "om"),
    ("Oriya", "or"),
    ("Ossetian", "os"),
    ("Punjabi", "pa"),
    ("Pali", "pi"),
    ("Persian", "fa"),
    ("Polish", "pl"),
    ("Pashto", "ps"),
    ("Portuguese", "pt"),
    ("Quechua", "qu"),
    ("Romansh", "rm"),
    ("Rundi", "rn"),
    ("Romanian", "ro"),
    ("Russian", "ru"),
    ("Sanskrit", "sa"),
    ("Sardinian", "sc"),
    ("Sindhi", "sd"),
    ("Northern Sami", "se"),
    ("Samoan", "sm"),
    ("Sango", "sg"),
    ("Serbian", "sr"),
    ("Gaelic (Scottish)", "gd"),
    ("Shona", "sn"),
    ("Sinhala", "si"),
    ("Slovak", "sk"),
    ("Slovenian", "sl"),
    ("Somali", "so"),
    ("Southern Sotho", "st"),
    ("Spanish", "es"),
    ("Sundanese", "su"),
    ("Swahili", "sw"),
    ("Swati", "ss"),
    ("Swedish", "sv"),
    ("Tamil", "ta"),
    ("Telugu", "te"),
    ("Tajik", "tg"),
    ("Thai", "th"),
    ("Tigrinya", "ti"),
    ("Tibetan", "bo"),
    ("Turkmen", "tk"),
    ("Tagalog", "tl"),
    ("Tswana", "tn"),
    ("Tonga", "to"),
    ("Turkish", "tr"),
    ("Tsonga", "ts"),
    ("Tatar", "tt"),
    ("Twi", "tw"),
    ("Tahitian", "ty"),
    ("Uighur", "ug"),
    ("Ukrainian", "uk"),
    ("Urdu", "ur"),
    ("Roman Urdu (Urdu in English letters)", "ur-Latn"),
    ("Uzbek", "uz"),
    ("Venda", "ve"),
    ("Vietnamese", "vi"),
    ("Volapük", "vo"),
    ("Walloon", "wa"),
    ("Welsh", "cy"),
    ("Wolof", "wo"),
    ("Western Frisian", "fy"),
    ("Xhosa", "xh"),
    ("Yiddish", "yi"),
    ("Yoruba", "yo"),
    ("Zhuang", "za"),
    ("Zulu", "zu"),
]


WEBSITE = [
    'Linkedin',
    'Facebook',
    'Instagram',
    'Youtube',
    'X (Twitter)',
    'Threads',
    'Tiktok',
    'Snapchat',
    'Pinterest',
    'Reddit',
    'Quora',
    'Discord',
    'Whatsapp',
    'Telegram',
    'Bluesky',
    'Mastodon',
    'BeReal',
    'Line',
    'Clubhouse',
    'Tumblr',
    'VK (VKontakte)',
    'WeChat (Moments)',
    'Weibo',
    'Viber',
    'Signal',
    'Nextdoor',
    'MeWe',
    'Truth Social',
    'Gettr',
    'Vero',
    'Nostr',
    'Lemmy',
    'Twitch',
    'Vimeo',
    'Dailymotion',
    'Kick (livestreaming)',
    'YouTube Shorts',
    'Instagram Reels',
    'TikTok LIVE',
    'Behance',
    'Dribbble',
    'ArtStation',
    'DeviantArt',
    '500px',
    'Flickr',
    'Unsplash contributor post',
    'GitHub README/profile',
    'GitLab profile',
    'Dev.to',
    'Hashnode',
    'Stack Overflow post',
    'Product Hunt launch post',
    'Hacker News (Show HN) post',
    'IndieHackers post',
    'AngelList/Wellfound',
    'Crunchbase update',
    'Glassdoor company update',
    'Xing (German LinkedIn)',
    'Fiverr gigs',
    'Upwork profile/proposal',
    'Freelancer.com profile',
    'PeoplePerHour listing',
    'Toptal profile',
    'Etsy shop update',
    'Shopify/store announcement',
    'Amazon listing blurb',
    'eBay listing',
    'Depop shop update',
    'Poshmark listing',
    'Alibaba storefront',
    'Walmart Marketplace listing',
    'Faire wholesale listing',
    'Medium',
    'Substack/Newsletter',
    'Google Business Profile',
    'WordPress blog post',
    'Ghost blog post',
    'Blogger post',
    'Quora Space post',
    'Slack community post',
    'Discourse forum post',
    'Facebook Group post',
    'LinkedIn Group post',
    'Circle community post',
    'Spotify podcast description',
    'Apple Podcasts show notes',
    'Anchor/Spotify for Podcasters post',
    'Clubhouse room description',
    'WhatsApp Status',
    'WhatsApp Business Broadcast',
    'Telegram Channel post',
    'SMS/text blast',
    'Email newsletter blast',
    'Yelp business update',
    'TripAdvisor listing update',
    'Google Reviews response',
    'KakaoTalk (Korea)',
    'Naver Blog (Korea)',
    'LINE VOOM (Japan)',
    'Zalo (Vietnam)',
    'OK.ru (Russia)',
    'Douyin (China)',
    'Xiaohongshu/RedNote (China)',
    'Kuaishou (China)',
    'Baidu Tieba (China)',
    'Sina Weibo (China)',
    'Bigo Live',
    'Sharechat (India)',
    'Koo (India)',
    'Moj (India)',
]

# WEBSITE count: 110

# Sensible default image canvas per platform.
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
    # --- newly added ---
    "Bluesky": (1200, 675),
    "Mastodon": (1200, 630),
    "BeReal": (1080, 1920),
    "Twitch": (1280, 720),
    "Vimeo": (1280, 720),
    "Behance": (1400, 1050),
    "Dribbble": (1600, 1200),
    "Product Hunt launch post": (1270, 760),
    "Nextdoor": (1200, 630),
}


def _aspect_label(width: int, height: int) -> str:
    """Rough aspect-ratio label per platform, used only to *describe* the
    desired framing to the image tool (Mistral's image agent doesn't take
    explicit width/height parameters, only prompt text)."""
    ratio = width / height
    if abs(ratio - 1) < 0.05:
        return "square (1:1)"
    if ratio > 1:
        return "landscape / horizontal, wide format"
    return "portrait / vertical, tall format"


# ---------------------------------------------------------------------------
# Sidebar Settings
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
    st.subheader("🖼️ Image generation (Mistral)")

    mistral_api_key = st.text_input(
        "Mistral API Key",
        value=os.environ.get("MISTRAL_API_KEY", ""),
        type="password",
        help="Get a key at https://console.mistral.ai. Required whenever "
             "you generate an image for a post.",
    )
    mistral_image_model = st.selectbox(
        "Mistral agent model",
        ["mistral-medium-latest", "mistral-large-latest", "mistral-small-latest"],
        index=0,
        help="The LLM that drives the image-generation agent.",
    )

    st.caption(
        "🖼️ Image generation runs through Mistral's Agents API "
        "(image_generation connector) and bills through your Mistral "
        "account."
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

# Input Configuration Selectors
col1, col2, col3 = st.columns(3)
with col1:
    category = st.selectbox("Post type", CATEGORIES)
with col2:
    tone = st.selectbox("Tone", TONES)
with col3:
    website = st.selectbox("Web platform target", WEBSITE, index=0)

topic = st.text_area(
    "Topic — what's the post about?",
    placeholder="e.g. I just shipped a new feature that reduced load times by 40%...",
    height=120,
)

# Optional product/item details - only relevant for e-commerce-style posts
# (product showcase, sale, listing blurb, etc.) but harmless to leave off
# for any other category.
include_product = st.checkbox("🛍️ This post is about a specific product/item", value=False)
product_name = ""
product_price = ""
product_features = ""
if include_product:
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        product_name = st.text_input("Product/Item name", placeholder="e.g. Aura Wireless Earbuds")
    with p_col2:
        product_price = st.text_input("Price (optional)", placeholder="e.g. $49.99 or Rs. 6,999")
    product_features = st.text_area(
        "Key features/selling points (optional)",
        placeholder="e.g. 30hr battery, active noise cancellation, waterproof, 2-year warranty",
        height=80,
    )

# Output Style Controls
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

# Generate multiple variants at once so the user can pick the best one
# instead of re-rolling one-at-a-time.
num_variants = st.slider(
    "🧪 Number of variants to generate",
    min_value=1,
    max_value=3,
    value=1,
    help="Generate several different takes on the same brief in one go, "
         "shown side-by-side in tabs, and pick your favorite.",
)

generate_image_toggle = st.checkbox(
    "🎨 Also generate a matching image for this post",
    value=False,
    help="Creates an AI image sized for the selected platform, based on the "
         "post's topic, tone, AND the actual generated post text. When "
         "generating multiple variants, the image is created for "
         "whichever variant you select as the keeper.",
)


def build_product_block(name: str, price: str, features: str) -> str:
    """Turn the optional product/item fields into a labelled block the
    prompt template can drop in as extra grounding. Returns an empty
    string when no product info was provided, so it's harmless to include
    unconditionally in the template."""
    name = (name or "").strip()
    price = (price or "").strip()
    features = (features or "").strip()
    if not (name or price or features):
        return ""
    lines = ["Product/Item details to reference in the post:"]
    if name:
        lines.append(f"- Name: {name}")
    if price:
        lines.append(f"- Price: {price}")
    if features:
        lines.append(f"- Key features/selling points: {features}")
    return "\n".join(lines)


def load_examples(category_name: str, n: int = 2):
    path = os.path.join(os.path.dirname(__file__), "linkedin_post_dataset.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

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

{product_block}

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


def build_image_prompt(topic: str, category: str, tone: str, post_text: str = "", aspect_label: str = "",
                        product_name: str = "") -> str:
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
    if product_name.strip():
        prompt += f" The central subject should be the product: {product_name.strip()}."
    if post_text.strip():
        # Keep this short - we only want the gist/keywords of the post as
        # visual grounding, not the literal caption rendered as an image.
        gist = " ".join(post_text.strip().split()[:40])
        prompt += f" The image should visually reflect the theme of this caption: {gist}"
    if aspect_label:
        prompt += f" Composition/framing: {aspect_label}."
    return prompt


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


def generate_image(prompt: str, mistral_key: str, mistral_model: str) -> bytes:
    """Generate an image via Mistral's Agents API (image_generation
    connector, Black Forest Labs FLUX1.1 [pro] Ultra under the hood).
    Mistral's image tool doesn't take explicit width/height - framing is
    steered through the prompt text instead. Returns raw image bytes."""
    if not mistral_key:
        raise ValueError("Mistral API key is required to generate images.")

    client, agent_id = get_mistral_image_agent(mistral_key, mistral_model)
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


def render_image_block(prompt: str, width: int, height: int, state_key: str, mistral_key: str, mistral_model: str) -> None:
    """Show the generated image (if any) for `state_key`, plus a
    regenerate button that rerolls."""
    if state_key in st.session_state:
        st.image(
            st.session_state[state_key],
            caption=f"AI-generated image ({width}x{height}) · Mistral",
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
                st.session_state[state_key] = generate_image(prompt, mistral_key, mistral_model)
                st.rerun()
            except Exception as e:
                st.error(f"Image generation failed: {e}")


def _make_image_for_entry(history_entry: dict, website: str, topic: str, category: str, tone: str, post_text: str,
                           mistral_api_key: str, mistral_image_model: str, product_name: str = "") -> None:
    """Generate and attach a matching image to a history entry in-place."""
    img_width, img_height = PLATFORM_IMAGE_SIZE.get(website, (1080, 1080))
    aspect_label = _aspect_label(img_width, img_height)
    image_prompt = build_image_prompt(topic, category, tone, post_text, aspect_label, product_name)
    state_key = f"post_image_{len(st.session_state.history)}"
    with st.spinner("Generating a matching image with Mistral..."):
        try:
            st.session_state[state_key] = generate_image(image_prompt, mistral_api_key, mistral_image_model)
            history_entry["image_key"] = state_key
            history_entry["image_prompt"] = image_prompt
        except Exception as e:
            st.error(f"Image generation failed: {e}")


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
    elif generate_image_toggle and not mistral_api_key:
        st.error("Please enter your Mistral API key in the sidebar to generate images.")
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

        product_block = build_product_block(product_name, product_price, product_features) if include_product else ""

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
                            "product_block": product_block,
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
                "product_name": product_name,
                "generate_image_toggle": generate_image_toggle,
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
        history_entry = {
            "platform": cfg["website"],
            "content": chosen_text,
            "topic": cfg["topic"],
            "category": cfg["category"],
            "tone": cfg["tone"],
            "product_name": cfg.get("product_name", ""),
            "image_key": None,
        }

        if cfg["generate_image_toggle"]:
            _make_image_for_entry(
                history_entry, cfg["website"], cfg["topic"], cfg["category"], cfg["tone"],
                chosen_text, cfg["mistral_api_key"], cfg["mistral_image_model"], cfg.get("product_name", ""),
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
            image_prompt = historical_post.get("image_prompt")
            if image_key or image_prompt:
                img_width, img_height = PLATFORM_IMAGE_SIZE.get(historical_post["platform"], (1080, 1080))
                if not image_key:
                    image_key = f"post_image_hist_{real_idx}"
                if not image_prompt:
                    aspect_label = _aspect_label(img_width, img_height)
                    image_prompt = build_image_prompt(
                        historical_post.get("topic", ""), historical_post.get("category", ""),
                        historical_post.get("tone", ""), historical_post["content"], aspect_label,
                        historical_post.get("product_name", ""),
                    )
                render_image_block(
                    image_prompt, img_width, img_height, image_key,
                    mistral_api_key, mistral_image_model,
                )
                historical_post["image_key"] = image_key
                historical_post["image_prompt"] = image_prompt
