from dc_custom_component.components.generators.image_generator import (
    DALLEImageGenerator,
)


class TestDALLEImageGenerator:
    def test_to_dict(self) -> None:
        generator = DALLEImageGenerator()
        data = generator.to_dict()
        assert data == {
            "type": "dc_custom_component.components.generators.image_generator.DALLEImageGenerator",
            "init_parameters": {
                "model": "dall-e-3",
                "quality": "standard",
                "size": "1024x1024",
                "response_format": "url",
                "api_key": {
                    "type": "env_var",
                    "env_vars": ["OPENAI_API_KEY"],
                    "strict": True,
                },
                "api_base_url": None,
                "organization": None,
            },
        }

    def test_from_dict(self) -> None:
        data = {
            "type": "dc_custom_component.components.generators.image_generator.DALLEImageGenerator",
            "init_parameters": {
                "model": "dall-e-3",
                "quality": "standard",
                "size": "1024x1024",
                "response_format": "url",
                "api_key": {
                    "type": "env_var",
                    "env_vars": ["OPENAI_API_KEY"],
                    "strict": True,
                },
                "api_base_url": None,
                "organization": None,
            },
        }
        generator = DALLEImageGenerator.from_dict(data)
        assert generator.model == "dall-e-3"
        assert generator.quality == "standard"
        assert generator.size == "1024x1024"
        assert generator.response_format == "url"
        assert generator.api_key.to_dict() == {
            "type": "env_var",
            "env_vars": ["OPENAI_API_KEY"],
            "strict": True,
        }
