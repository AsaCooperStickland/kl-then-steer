from llmtuner.data.template import register_template
from lat.utils import system_prompt

register_template(
    name="llama2simple",
    prefix=[
        "<<SYS>>\n{{system}}\n<</SYS>>\n\n"
    ],
    prompt=[
        "[INST] {{query}} [/INST]"
    ],
    system=(
        system_prompt,
    ),
    sep=[]
)