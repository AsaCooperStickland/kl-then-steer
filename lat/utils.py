system_prompt = "You are a helpful, honest and concise assistant."
formatted_system_prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
data_path = "datasets/refusal/refusal_data_A_B.json"
jailbreaks_path = "datasets/refusal/jailbreaks.json"

criminal_concepts = ["crime", "criminal", "shoplifting", "murder", "fraud", "spearphishing", "burglary", "blackmail", "prison",
                     "illegal", "armed_robbery", "hacking", "social_engineering", "terrorism", "weapons", "guns", "knuckledusters",
                     "stabbing", "drug_dealing", "gangsters", "mafia", "shooting", "drug_trafficking", "insider_trading", "phishing", "sql_injection"
                     "scam"]
                
bad_concepts = [
"evil", "bratty", "sleazy", "nasty", "cold and calculating", "scheming", "disgusting", "amoral", 
"exploitative", "mean", "know-it-all", "deceitful", "manipulative", "ruthless", "vindictive", "callous", 
"unscrupulous", "malicious", "greedy", "heartless", "cruel", "vengeful", "selfish", "unethical", "treacherous", 
"violent", "sadistic", "merciless", "cunning", "opportunistic", "tyrannical", "envious", "corrupt", "untrustworthy",
"arrogant", "bullying", "cynical", "deceptive", "egotistical", "fraudulent", "harsh", "insidious", "jealous", 
"malevolent", "narcissistic", "obnoxious", "pessimistic", "quarrelsome", "rebellious", "spiteful", "troublesome",
"unreliable", "vain", "wicked", "xenophobic", "yelling", "zealous", "abrasive", "belligerent", "coercive", 
"despotic", "egotistic", "fierce", "grudging", "impatient", "jaded", "loathsome", "malignant", "nefarious",
"oppressive", "pernicious", "resentful", "severe", "tyrannous", "unforgiving", "vicious", "wrathful", "argumentative",
"boastful", "contemptuous", "distrustful", "erratic", "fickle", "grumpy", "hypocritical", "inflexible", "judgmental",
"lethargic", "moody", "nonchalant", "overbearing", "provocative", "rigid", "stubborn", "tactless", "unapologetic",
"volatile", "willful", "adamant", "bitter", "crass", "disdainful", "explosive", "flippant", "heedless", "ignorant", 
"imprudent", "knavish", "loquacious", "mischievous", "neglectful", "obstinate", "petty", "rash", "sullen", "tumultuous",
"unyielding", "verbose", "whimsical", "acerbic", "blunt", "combative", "derisive", "excessive", "fanatical", "hasty",
"imperious", "impulsive", "kleptomaniac", "languid", "morose", "noxious", "ostentatious", "pedantic", "reproachful", 
"sarcastic", "tempestuous", "unpredictable", "venomous", "wily", "antagonistic", "bellicose", "condescending", 
"duplicitous", "elitist", "frenetic", "haughty", "indolent", "insolent", "licentious", "misanthropic", "neurotic", 
"overcritical", "pompous", "reticent", "sanctimonious", "thrifty", "underhanded", "vulgar", "wary"
]

neutral_concepts = [
    "water", "sunshine", "curiosity", "tennis", "exciting",
    "music", "gardening", "reading", "hiking", "painting",
    "baking", "birdwatching", "yoga", "stargazing", "knitting",
    "photography", "dancing", "camping", "snowboarding", "surfing",
    "cooking", "fishing", "journaling", "meditation", "pottery",
    "taekwondo", "volunteering", "crocheting", "pilates", "traveling",
]
neutral_concepts += [
    "rainbow", "clouds", "mountains", "oceans", "forests",
    "deserts", "cities", "villages", "islands", "rivers",
    "sunsets", "sunrises", "stars", "planets", "galaxies",
    "animals", "birds", "fish", "insects", "flowers",
    "trees", "fruits", "vegetables", "minerals", "metals",
    "colors", "sounds", "textures", "smells", "tastes",
    "numbers", "letters", "words", "languages", "cultures",
    "emotions", "thoughts", "dreams", "memories", "ideas",
    "energy", "time", "space", "gravity", "light",
    "darkness", "seasons", "weather", "temperatures", "directions",
    "shapes", "sizes", "distances", "speeds", "frequencies",
    "patterns", "rhythms", "harmonies", "melodies", "noises",
    "silence", "echoes", "shadows", "reflections", "illusions",
    "theories", "principles", "laws", "hypotheses",
    "questions", "answers", "problems", "solutions", "discoveries",
    "inventions", "creations", "artifacts", "relics", "treasures",
    "myths", "legends", "stories", "tales", "fables",
    "jokes", "riddles", "puzzles", "games", "sports",
    "dances", "rituals", "ceremonies", "traditions", "customs"
]

neutral_concepts += [
    "surprise", "trust", "anticipation", "joy", "sorrow", "envy", "sympathy", "hope", "despair", "shame", "guilt", "curiosity", "boredom",
    "longing", "regret", "affection", "amazement",
    "excitement", "grief", "passion", "restlessness", "tenderness",
    "uncertainty", "wistfulness", "zeal", "acquiescence", "bliss",
    "dejection", "elation", "fascination", "indignation", "jubilation",
    "mirth", "rapture", "reverence", "skepticism", "vulnerability",
    "wonder", "yearning", "zest", "empathy"
]
