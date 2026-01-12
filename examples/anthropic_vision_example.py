"""Example demonstrating Anthropic vision capabilities with thinllm."""

import os

from thinllm.config import LLMConfig
from thinllm.core import llm
from thinllm.messages import InputImageBlock, InputTextBlock, UserMessage


def main() -> None:
    """Demonstrate vision capabilities with Anthropic."""
    # Configure Anthropic LLM
    config = LLMConfig(
        provider="anthropic",
        model="claude-sonnet-4-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    # Example 1: Using image URL
    print("=" * 50)
    print("Example 1: Image from URL")
    print("=" * 50)

    messages_url = [
        UserMessage(
            content=[
                InputImageBlock(
                    image_url="https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg"
                ),
                InputTextBlock(text="Describe this image in one sentence."),
            ]
        )
    ]

    response_url = llm(messages=messages_url, config=config)
    print(f"Response: {response_url.content}\n")

    # Example 2: Using base64-encoded image from file
    print("=" * 50)
    print("Example 2: Image from file (base64)")
    print("=" * 50)

    # You can load an image from a file like this:
    # image_block = InputImageBlock.from_file("path/to/image.jpg")
    # For demonstration, we'll create a simple image block with sample data

    import base64

    # For this example, we'll use a small transparent PNG (1x1 pixel)
    # In a real scenario, you would load your actual image file
    small_png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )

    messages_base64 = [
        UserMessage(
            content=[
                InputImageBlock(
                    image_bytes=small_png,
                    mimetype="image/png",
                ),
                InputTextBlock(text="What color is this image?"),
            ]
        )
    ]

    response_base64 = llm(messages=messages_base64, config=config)
    print(f"Response: {response_base64.content}\n")

    # Example 3: Multiple images
    print("=" * 50)
    print("Example 3: Multiple images")
    print("=" * 50)

    messages_multiple = [
        UserMessage(
            content=[
                InputTextBlock(text="Image 1:"),
                InputImageBlock(
                    image_url="https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg"
                ),
                InputTextBlock(text="Image 2:"),
                InputImageBlock(
                    image_url="https://upload.wikimedia.org/wikipedia/commons/b/b5/Iridescent.green.sweat.bee1.jpg"
                ),
                InputTextBlock(text="What are the main differences between these two insects?"),
            ]
        )
    ]

    response_multiple = llm(messages=messages_multiple, config=config)
    print(f"Response: {response_multiple.content}\n")

    # Example 4: Loading image from file
    print("=" * 50)
    print("Example 4: Loading image from file")
    print("=" * 50)

    # Uncomment this when you have an actual image file:
    # image_block = InputImageBlock.from_file("path/to/your/image.jpg")
    # messages_file = [
    #     UserMessage(
    #         content=[
    #             image_block,
    #             InputTextBlock(text="Analyze this image."),
    #         ]
    #     )
    # ]
    # response_file = llm(messages=messages_file, config=config)
    # print(f"Response: {response_file.content}\n")

    print("Note: Uncomment Example 4 when you have an image file to test.")


if __name__ == "__main__":
    main()
