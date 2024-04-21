"""
Using an LLM to explain feature activation patterns.
Inspired by:
- https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-automated
- https://github.com/openai/automated-interpretability/
"""

import os
import pandas as pd

from evaluation.autoencoder.automated_interpretability.activations import (
    Activation,
    select_activation_samples,
)
from openai import AsyncOpenAI

if not (OPENAI_KEY := os.getenv("OPENAI_KEY")):
    raise EnvironmentError(
        "Environment variable `OPENAI_KEY` required for automated interpretability."
    )

openai_client = AsyncOpenAI(api_key=OPENAI_KEY)


TOKEN_ACTIVATION_SYSTEM_PROMPT = """
We're analysing the neurons of a neural network. You will be provided with independent (i.e. not listed in sequence) samples of tokens the corresponding neuron activations.
Look at the tokens the neuron activates for, and summarize what the neuron is looking for. The token activation format is <quoted token><tab><activation strength>.
A neuron finding it's looking for is represented by a non-zero activation value. The higher the activation value, the stronger the match.

Think hard about which tokens cause the highest activations, and describe the activation pattern.

Don't mention any examples of tokens.

Complete the sentence provided by the user.
""".strip()


TOKEN_ACTIVATION_EXAMPLE_INTERACTION = [
    {
        "role": "user",
        "content": """
Feature #1002
Activations:
<sample start>
'9'\t9.0
'"'\t9.0
'`'\t0.0
's'\t0.0
'.'\t1.0
'\t\t   '\t1.0
'E'\t2.0
'.'\t2.0
'\t\t   '\t3.0
"',"\t3.0
'rel'\t4.0
'sh'\t4.0
'<s>'\t5.0
'schema'\t   5.0
"'"\t6.0
'a'\t6.0
'a'\t7.0
'Oliver'\t   7.0
'1'\t8.0
'H'\t8.0
'\t '\t   3.0
'was'\t0.0
't'\t0.0
'\n'\t0.0
'Des'\t1.0
'"'\t4.0
'9'\t2.0
'9'\t0.0
'9'\t3.0
'9'\t7.0
'"'\t0.0
'"'\t3.0
'"'\t5.0
'"'\t2.0
'9'\t0.0
<sample end>

Explain the behaviour of Feature #1002, without mentioning specific examples of tokens, by completing the sentence "This feature looks mainly for"
        """,
    },
    {"role": "assistant", "content": "single-digit numbers and quotation marks."},
]


async def explain_feature_token_activation(
    df_feature_information: pd.DataFrame, feature_number: int
) -> str:
    """Explain feature by its token activation data.

    Args:
        df_feature_information (pd.DataFrame): DataFrame containing feature information (i.e. activation data).
        feature_number (int): Feature number.

    Returns:
        str: Explanation of feature.
    """
    # It is not polite to hardcode values ...
    return await openai_explanation(
        model_name="gpt-4-turbo",
        system_prompt=TOKEN_ACTIVATION_SYSTEM_PROMPT,
        messages=[
            *TOKEN_ACTIVATION_EXAMPLE_INTERACTION,
            {
                "role": "user",
                "content": TokenActivation.explanation_prompt_for_feature(
                    df=df_feature_information,
                    feature_number=feature_number,
                    num_samples=1,
                ),
            },
        ],
        temperature=0,
        top_p=1.0,
    )


async def openai_explanation(
    model_name: str,
    system_prompt: str,
    messages: list[dict[str, str]],
    temperature: float,
    top_p: float,
) -> str:
    """Get OpenAI model explanations from prompt.

    Args:
        model_name (str): Name of model to use.
        prompt (str): Prompt to get explanations from.
        messages (list[dict[str, str]]): Messages in conversation.
        temperature (float): Temperature of nucleus sampling generation.
        top_p (float): Top-P value of sampling.

    Returns:
        str: Explanation extracted from model response.
    """
    final_messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        *messages,
    ]
    response = await openai_client.chat.completions.create(
        model=model_name, messages=final_messages, temperature=temperature, top_p=top_p
    )
    response_message: str = response.choices[0].message.content
    return response_message


def activation_samples_overview(activations: list[Activation]) -> str:
    """Get overview of activation samples.

    Args:
        activations (list[ActivationSample]): Activation samples to get overview of.

    Returns:
        str: Overview string.
    """
    sample_listing = "\n".join(
        [
            f"{activation.token!r}\t{activation.activation_quantized}"
            for activation in activations
        ]
    )
    return f"<sample start>\n{sample_listing}\n<sample end>"


class TokenActivation:
    """Explain feature based on token activation pattern."""

    @staticmethod
    def explanation_prompt_for_feature(
        df: pd.DataFrame,
        feature_number: int,
        num_samples: int,
    ) -> str:
        """Get token explanation prompt from feature.

        Args:
            df (pd.DataFrame): DataFrame containing the columns activation data.
            feature_number (int): Which feature to select samples for.
            num_samples (int): Number of samples of activations to provide in prompt.

        Returns:
            str: Prompt to ask nicely for explanation of feature.
        """
        activation_samples: list[list[Activation]] = [
            select_activation_samples(df=df, feature_number=feature_number)
            for _ in range(num_samples)
        ]

        prompt: str = TokenActivation.feature_explanation_prompt_from_samples(
            feature_number=feature_number, activation_samples=activation_samples
        )

        return prompt

    @staticmethod
    def feature_explanation_prompt_from_samples(
        feature_number: int, activation_samples: list[list[Activation]]
    ) -> str:
        """Get feature explanation prompt from token activation samples.

        Args:
            feature_number (int): Feature's number.
            activations (list[list[Activation]]): List of token activation samples.

        Returns:
            str: Prompt to get explanations.
        """
        activation_overviews = "\n".join(
            [
                activation_samples_overview(activations=sample)
                for sample in activation_samples
            ]
        )

        prompt: str = (
            f"Feature #{feature_number}\n"
            f"Activations:\n{activation_overviews}\n\n"
            f'Explain the behaviour of Feature #{feature_number}, without mentioning specific examples of tokens, by completing the sentence "This feature looks mainly for"'
        )

        return prompt
