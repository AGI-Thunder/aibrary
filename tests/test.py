from unittest.mock import patch

from aibrary.resources.aibrary_wrapper import AiBrary


def test_chat_completion_create():
    with patch("aibrary.chat.completions.create") as mock_create:
        # Define the mock return value
        mock_create.return_value = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I'm doing great, thank you!",
                    }
                }
            ]
        }

        # Call the method to test
        result = AiBrary().chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "How are you today?"},
                {"role": "assistant", "content": "I'm doing great, thank you!"},
            ],
            temperature=0.7,
        )

        # Assertions to verify behavior
        mock_create.assert_called_once_with(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "How are you today?"},
                {"role": "assistant", "content": "I'm doing great, thank you!"},
            ],
            temperature=0.7,
        )

        assert "choices" in result
        assert (
            result["choices"][0]["message"]["content"] == "I'm doing great, thank you!"
        )
