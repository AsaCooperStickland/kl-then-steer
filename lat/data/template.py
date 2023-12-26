from copy import deepcopy
from typing import TYPE_CHECKING, Dict, List, Union, Optional

from llmtuner.extras.logging import get_logger
from llmtuner.data.template import register_template, Template, Llama2Template
from lat.utils import system_prompt

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

logger = get_logger(__name__)


templates: Dict[str, Template] = {}


def register_template(
    name: str,
    prefix: List[Union[str, Dict[str, str]]],
    prompt: List[Union[str, Dict[str, str]]],
    system: str,
    sep: List[Union[str, Dict[str, str]]],
    stop_words: Optional[List[str]] = [],
    use_history: Optional[bool] = True,
    efficient_eos: Optional[bool] = False,
) -> None:
    template_class = Llama2Template if name.startswith("llama2") else Template
    templates[name] = template_class(
        prefix=prefix,
        prompt=prompt,
        system=system,
        sep=sep,
        stop_words=stop_words,
        use_history=use_history,
        efficient_eos=efficient_eos,
    )


def get_template_and_fix_tokenizer(
    name: str,
    tokenizer: "PreTrainedTokenizer"
) -> Template:
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = "<|endoftext|>"
        logger.info("Add eos token: {}".format(tokenizer.eos_token))

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Add pad token: {}".format(tokenizer.pad_token))

    if name is None: # for pre-training
        return None

    template = templates.get(name, None)
    assert template is not None, "Template {} does not exist.".format(name)
    tokenizer.add_special_tokens(
        dict(additional_special_tokens=template.stop_words),
        replace_additional_special_tokens=False
    )
    # Should figure this out but hopefully works for now https://github.com/hiyouga/LLaMA-Factory/commit/5af8841c4f6c97df522d2cf4e283d5ef0af21a18
    # stop_words = deepcopy(template.stop_words)
    # if template.replace_eos:
    #     if not stop_words:
    #         raise ValueError("Stop words are required to replace the EOS token.")

    #     tokenizer.eos_token = stop_words.pop(0)
    #     logger.info("Replace eos token: {}".format(tokenizer.eos_token))

    # if stop_words:
    #     tokenizer.add_special_tokens(
    #         dict(additional_special_tokens=stop_words),
    #         replace_additional_special_tokens=False
    #     )
    #     logger.info("Add {} to stop words.".format(",".join(stop_words)))

    return template


register_template(
    name="llama2chatsimple",
    prefix=[
        "<<SYS>>\n{{system}}\n<</SYS>>\n\n"
    ],
    prompt=[
        "[INST] {{query}} [/INST]"
    ],
    system=system_prompt,
    sep=[]
)
