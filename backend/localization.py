"""Shared response-language policy for model-generated explanations."""

from __future__ import annotations


SUPPORTED_RESPONSE_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
    "kn": "Kannada",
    "bn": "Bengali",
    "mr": "Marathi",
    "gu": "Gujarati",
    "ml": "Malayalam",
    "pa": "Punjabi (Gurmukhi)",
    "or": "Odia",
    "ur": "Urdu",
}


_FALLBACKS = {
    "en": {
        "reasoning": "AI review was unavailable. This verdict uses the local classifier and retrieval signals.",
        "action": "Use caution and request human review.",
        "incremental": "Incremental local classifier and hybrid retrieval assessment.",
    },
    "hi": {
        "reasoning": "AI समीक्षा उपलब्ध नहीं थी। यह निर्णय स्थानीय वर्गीकरण और पुनर्प्राप्ति संकेतों पर आधारित है।",
        "action": "सावधानी बरतें और मानवीय समीक्षा का अनुरोध करें।",
        "incremental": "स्थानीय वर्गीकरण और हाइब्रिड पुनर्प्राप्ति का क्रमिक आकलन।",
    },
    "te": {
        "reasoning": "AI సమీక్ష అందుబాటులో లేదు. ఈ నిర్ణయం స్థానిక వర్గీకరణ మరియు సమాచార పునరుద్ధరణ సంకేతాలపై ఆధారపడింది.",
        "action": "జాగ్రత్త వహించి మానవ సమీక్షను కోరండి.",
        "incremental": "స్థానిక వర్గీకరణ మరియు హైబ్రిడ్ సమాచార పునరుద్ధరణ యొక్క దశలవారీ అంచనా.",
    },
    "ta": {
        "reasoning": "AI மதிப்பாய்வு கிடைக்கவில்லை. இந்த முடிவு உள்ளூர் வகைப்படுத்தல் மற்றும் மீட்டெடுப்பு சிக்னல்களைப் பயன்படுத்துகிறது.",
        "action": "எச்சரிக்கையுடன் இருந்து மனித மதிப்பாய்வைக் கோரவும்.",
        "incremental": "உள்ளூர் வகைப்படுத்தல் மற்றும் கலப்பு மீட்டெடுப்பின் படிப்படியான மதிப்பீடு.",
    },
    "kn": {
        "reasoning": "AI ಪರಿಶೀಲನೆ ಲಭ್ಯವಿರಲಿಲ್ಲ. ಈ ತೀರ್ಪು ಸ್ಥಳೀಯ ವರ್ಗೀಕರಣ ಮತ್ತು ಮರುಪಡೆಯುವಿಕೆ ಸಂಕೇತಗಳನ್ನು ಬಳಸುತ್ತದೆ.",
        "action": "ಎಚ್ಚರಿಕೆ ವಹಿಸಿ ಮಾನವ ಪರಿಶೀಲನೆಯನ್ನು ಕೋರಿ.",
        "incremental": "ಸ್ಥಳೀಯ ವರ್ಗೀಕರಣ ಮತ್ತು ಹೈಬ್ರಿಡ್ ಮರುಪಡೆಯುವಿಕೆಯ ಹಂತ ಹಂತದ ಮೌಲ್ಯಮಾಪನ.",
    },
    "bn": {
        "reasoning": "AI পর্যালোচনা পাওয়া যায়নি। এই সিদ্ধান্ত স্থানীয় শ্রেণিবিন্যাস ও তথ্য পুনরুদ্ধার সংকেতের উপর ভিত্তি করে।",
        "action": "সতর্ক থাকুন এবং মানবিক পর্যালোচনার অনুরোধ করুন।",
        "incremental": "স্থানীয় শ্রেণিবিন্যাস ও হাইব্রিড তথ্য পুনরুদ্ধারের ধাপে ধাপে মূল্যায়ন।",
    },
    "mr": {
        "reasoning": "AI पुनरावलोकन उपलब्ध नव्हते. हा निर्णय स्थानिक वर्गीकरण आणि माहिती पुनर्प्राप्ती संकेतांवर आधारित आहे.",
        "action": "सावध राहा आणि मानवी पुनरावलोकनाची विनंती करा.",
        "incremental": "स्थानिक वर्गीकरण आणि हायब्रिड माहिती पुनर्प्राप्तीचे टप्प्याटप्प्याने मूल्यांकन.",
    },
    "gu": {
        "reasoning": "AI સમીક્ષા ઉપલબ્ધ નહોતી. આ નિર્ણય સ્થાનિક વર્ગીકરણ અને માહિતી પુનઃપ્રાપ્તિ સંકેતો પર આધારિત છે.",
        "action": "સાવચેત રહો અને માનવીય સમીક્ષાની વિનંતી કરો.",
        "incremental": "સ્થાનિક વર્ગીકરણ અને હાઇબ્રિડ માહિતી પુનઃપ્રાપ્તિનું તબક્કાવાર મૂલ્યાંકન.",
    },
    "ml": {
        "reasoning": "AI അവലോകനം ലഭ്യമായിരുന്നില്ല. ഈ തീരുമാനം പ്രാദേശിക വർഗ്ഗീകരണവും വിവര വീണ്ടെടുക്കൽ സൂചനകളും ഉപയോഗിക്കുന്നു.",
        "action": "ജാഗ്രത പാലിച്ച് മനുഷ്യ അവലോകനം അഭ്യർഥിക്കുക.",
        "incremental": "പ്രാദേശിക വർഗ്ഗീകരണത്തിന്റെയും ഹൈബ്രിഡ് വിവര വീണ്ടെടുക്കലിന്റെയും ഘട്ടംഘട്ടമായ വിലയിരുത്തൽ.",
    },
    "pa": {
        "reasoning": "AI ਸਮੀਖਿਆ ਉਪਲਬਧ ਨਹੀਂ ਸੀ। ਇਹ ਫੈਸਲਾ ਸਥਾਨਕ ਵਰਗੀਕਰਨ ਅਤੇ ਜਾਣਕਾਰੀ ਪ੍ਰਾਪਤੀ ਸੰਕੇਤਾਂ ਉੱਤੇ ਆਧਾਰਿਤ ਹੈ।",
        "action": "ਸਾਵਧਾਨ ਰਹੋ ਅਤੇ ਮਨੁੱਖੀ ਸਮੀਖਿਆ ਦੀ ਬੇਨਤੀ ਕਰੋ।",
        "incremental": "ਸਥਾਨਕ ਵਰਗੀਕਰਨ ਅਤੇ ਹਾਈਬ੍ਰਿਡ ਜਾਣਕਾਰੀ ਪ੍ਰਾਪਤੀ ਦਾ ਪੜਾਅਵਾਰ ਮੁਲਾਂਕਣ।",
    },
    "or": {
        "reasoning": "AI ସମୀକ୍ଷା ଉପଲବ୍ଧ ନଥିଲା। ଏହି ନିଷ୍ପତ୍ତି ସ୍ଥାନୀୟ ବର୍ଗୀକରଣ ଓ ସୂଚନା ପୁନରୁଦ୍ଧାର ସଙ୍କେତ ଉପରେ ଆଧାରିତ।",
        "action": "ସତର୍କ ରୁହନ୍ତୁ ଏବଂ ମାନବ ସମୀକ୍ଷା ଅନୁରୋଧ କରନ୍ତୁ।",
        "incremental": "ସ୍ଥାନୀୟ ବର୍ଗୀକରଣ ଓ ହାଇବ୍ରିଡ୍ ସୂଚନା ପୁନରୁଦ୍ଧାରର ପର୍ଯ୍ୟାୟକ୍ରମିକ ମୂଲ୍ୟାୟନ।",
    },
    "ur": {
        "reasoning": "AI جائزہ دستیاب نہیں تھا۔ یہ فیصلہ مقامی درجہ بندی اور معلوماتی بازیافت کے اشاروں پر مبنی ہے۔",
        "action": "احتیاط کریں اور انسانی جائزے کی درخواست کریں۔",
        "incremental": "مقامی درجہ بندی اور ہائبرڈ معلوماتی بازیافت کا مرحلہ وار جائزہ۔",
    },
}


def normalize_language(language: str | None) -> str:
    """Return a supported base locale, defaulting to English."""
    locale = (language or "en").strip().lower().split("-")[0]
    return locale if locale in SUPPORTED_RESPONSE_LANGUAGES else "en"


def model_language_instruction(language: str | None) -> str:
    """Tell an LLM which values to localize without translating enums."""
    locale = normalize_language(language)
    name = SUPPORTED_RESPONSE_LANGUAGES[locale]
    return (
        f"Write every human-facing explanation, finding, indicator, summary, and recommended action "
        f"in {name} (locale {locale}). Keep JSON keys and the canonical verdict, risk_level, severity, "
        "and false_positive_risk enum values exactly in English. Do not translate quoted evidence "
        "from the original conversation."
    )


def localized_fallback(language: str | None, key: str) -> str:
    locale = normalize_language(language)
    return _FALLBACKS[locale][key]
