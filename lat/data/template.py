from copy import deepcopy
from typing import TYPE_CHECKING, Dict, List, Union, Optional

from llmtuner.extras.logging import get_logger
from llmtuner.data.template import Template, Llama2Template
from llmtuner.data.formatter import EmptyFormatter, FunctionFormatter, StringFormatter, ToolFormatter
from lat.utils import system_prompt

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer
    from llmtuner.data.formatter import Formatter

logger = get_logger(__name__)


templates: Dict[str, Template] = {}


def _register_template(
    name: str,
    format_user: Optional["Formatter"] = None,
    format_assistant: Optional["Formatter"] = None,
    format_system: Optional["Formatter"] = None,
    format_function: Optional["Formatter"] = None,
    format_observation: Optional["Formatter"] = None,
    format_tools: Optional["Formatter"] = None,
    format_separator: Optional["Formatter"] = None,
    default_system: Optional[str] = "",
    stop_words: Optional[List[str]] = [],
    efficient_eos: Optional[bool] = False,
    replace_eos: Optional[bool] = False,
    force_system: Optional[bool] = False,
) -> None:
    eos_slots = [] if efficient_eos else [{"eos_token"}]
    template_class = Llama2Template if name.startswith("llama2") else Template
    default_user_formatter = StringFormatter(slots=["{{content}}"])
    default_assistant_formatter = StringFormatter(slots=["{{content}}"] + eos_slots)
    default_function_formatter = FunctionFormatter(slots=["Action: {{name}}\nAction Input: {{arguments}}"] + eos_slots)
    default_tool_formatter = ToolFormatter(tool_format="default")
    default_separator_formatter = EmptyFormatter()
    templates[name] = template_class(
        format_user=format_user or default_user_formatter,
        format_assistant=format_assistant or default_assistant_formatter,
        format_system=format_system or default_user_formatter,
        format_function=format_function or default_function_formatter,
        format_observation=format_observation or format_user or default_user_formatter,
        format_tools=format_tools or default_tool_formatter,
        format_separator=format_separator or default_separator_formatter,
        default_system=default_system,
        stop_words=stop_words,
        efficient_eos=efficient_eos,
        replace_eos=replace_eos,
        force_system=force_system,
    )


def _add_or_replace_eos_token(tokenizer: "PreTrainedTokenizer", eos_token: str) -> None:
    is_added = tokenizer.eos_token_id is None
    is_oov = eos_token not in tokenizer.get_vocab()
    tokenizer.add_special_tokens({"eos_token": eos_token})

    if is_added:
        logger.info("Add eos token: {}".format(tokenizer.eos_token))
    else:
        logger.info("Replace eos token: {}".format(tokenizer.eos_token))

    if is_oov:
        logger.warning("New tokens have been added, make sure `resize_vocab` is True.")


def get_template_and_fix_tokenizer(
    tokenizer: "PreTrainedTokenizer",
    name: Optional[str] = None,
) -> Template:
    if name is None:
        template = templates["vanilla"]  # placeholder
    else:
        template = templates.get(name, None)
        if templates is None:
            raise ValueError("Template {} does not exist.".format(name))

    stop_words = template.stop_words
    if template.replace_eos:
        if not stop_words:
            raise ValueError("Stop words are required to replace the EOS token.")

        _add_or_replace_eos_token(tokenizer, eos_token=stop_words[0])
        stop_words = stop_words[1:]

    if tokenizer.eos_token_id is None:
        _add_or_replace_eos_token(tokenizer, eos_token="<|endoftext|>")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Add pad token: {}".format(tokenizer.pad_token))

    if stop_words:
        tokenizer.add_special_tokens(
            dict(additional_special_tokens=stop_words), replace_additional_special_tokens=False
        )
        logger.info("Add {} to stop words.".format(",".join(stop_words)))

    return template


_register_template(
    name="alpaca",
    format_user=StringFormatter(slots=["### Instruction:\n{{content}}\n\n### Response:\n"]),
    format_separator=EmptyFormatter(slots=["\n\n"]),
    default_system=(
        "Below is an instruction that describes a task. " "Write a response that appropriately completes the request."
    ),
)

_register_template(
    name="llama2",
    format_user=StringFormatter(slots=[{"bos_token"}, "[INST] {{content}} [/INST]"]),
    format_system=StringFormatter(slots=["<<SYS>>\n{{content}}\n<</SYS>>\n\n"]),
    default_system=(
        "You are a helpful, respectful and honest assistant. "
        "Always answer as helpfully as possible, while being safe. "
        "Your answers should not include any harmful, unethical, "
        "racist, sexist, toxic, dangerous, or illegal content. "
        "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, "
        "explain why instead of answering something not correct. "
        "If you don't know the answer to a question, please don't share false information."
    ),
)

_register_template(
    name="llama2chatsimple",
    format_user=StringFormatter(slots=[{"bos_token"}, "[INST] {{content}} [/INST]"]),
    format_system=StringFormatter(slots=["<<SYS>>\n{{content}}\n<</SYS>>\n\n"]),
    default_system=system_prompt,
)

_register_template(
    name="llama2_empty_system_prompt",
    format_user=StringFormatter(slots=[{"bos_token"}, "[INST] {{content}} [/INST]"]),
    format_system=StringFormatter(slots=["<<SYS>>\n{{content}}\n<</SYS>>\n\n"]),
    default_system="",
)

_register_template(
    name="llama2_no_system_prompt",
    format_user=StringFormatter(slots=[{"bos_token"}, "[INST] {{content}} [/INST]"]),
    format_system=StringFormatter(slots=["{{content}}"]),
    default_system="",
)
