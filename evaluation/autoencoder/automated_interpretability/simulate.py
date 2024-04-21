"""Evaluate explanations by having GPT-4 Turbo simulate activations."""

import pandas as pd

from evaluation.autoencoder.automated_interpretability.explain import openai_explanation


TOKEN_ACTIVATION_SIMULATION_SYSTEM_PROMPT = """
Your task is to score tokens based on a description of which tokens usually score high.
You're provided with the description which tokens generally score high, and the token you need to provide a score for.

Scores range from 0 to 9. The higher the score value, the stronger the token fits the description of the tokens that usually score high.

Only reply with one of the following numbers, and nothing else: 0 1 2 3 4 5 6 7 8 9
""".strip()

TOKEN_ACTIVATION_SIMULATION_EXAMPLE_INTERACTION = [
    {
        "role": "user",
        "content": """
Description: \"single-digit numbers and quotation marks\"
Token: '9'
        """.strip(),
    },
    {"role": "assistant", "content": "9"},
    {
        "role": "user",
        "content": """
Description: \"spaces and periods, with a strong emphasis on sequences of multiple spaces.\"
Token: ' '
        """.strip(),
    },
    {"role": "assistant", "content": "6"},
    {
        "role": "user",
        "content": """
Description: \"spaces and periods, with a strong emphasis on sequences of multiple spaces.\"
Token: '        '
        """.strip(),
    },
    {"role": "assistant", "content": "9"},
]


async def token_activation_by_explanation(
    explanation: str, token: str, model_name: str, temperature: float, top_p: float
) -> int:
    """Simulate activation using GPT-4 Turbo to get activation for token.

    Args:
        explanation (str): Explanation of how to activate for token.
        token (str): Token to get activation for.
        model_name (str): Name of OpenAI completion model to use.
        temperature (float): Temperature of nucleus sampling generation.
        top_p (float): Top-P value of sampling.

    Returns:
        int: Quantized activation in range 0-9.
    """
    scoring_prompt: str = f'Description: "{explanation}"' f"Token: {token!r}"

    _, response = await openai_explanation(
        model_name=model_name,
        system_prompt=TOKEN_ACTIVATION_SIMULATION_SYSTEM_PROMPT,
        messages=[
            *TOKEN_ACTIVATION_SIMULATION_EXAMPLE_INTERACTION,
            {"role": "user", "content": scoring_prompt},
        ],
        temperature=temperature,
        top_p=top_p,
        logprobs=True,
        top_logprobs=10,
    )

    valid_score_logprobs: dict[str, float] = {
        logprob.token: logprob.logprob
        for logprob in response.choices[0].logprobs.content[0].top_logprobs
        if logprob.token in map(str, range(10))
    }

    selected_score: int = int(max(valid_score_logprobs, key=valid_score_logprobs.get))
    return selected_score


# Capturing closure to unquantize activations for feature.
def unquantize_activation(
    quantized_activation: float,
    df_feature_information: pd.DataFrame,
    feature_number: int,
) -> pd.Series:
    """Un-quantize the quantized activations based on original activations.

    Args:
        quantized_activation (float): Quantized activation in range [0,9].

    Returns:
        pd.Series: Approximated unquantized activations.
    """
    original_activations = df_feature_information[
        df_feature_information["feature"] == feature_number
    ]["activation"]
    approx_normalized_activations = (quantized_activation + 0.5) / 10
    return (
        approx_normalized_activations
        * (original_activations.max() - original_activations.min())
    ) + original_activations.min()
