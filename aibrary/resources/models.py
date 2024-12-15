class Model:
    def __init__(
        self,
        provider: str,
        category: str,
        model_name: str,
        price_per_input_unit: float,
        price_per_output_unit: float,
        pricing_input_unit_type: str,
        pricing_output_unit_type: str,
        status: str,
    ):
        self.provider = provider
        self.category = category
        self.model_name = model_name
        self.price_per_input_unit = price_per_input_unit
        self.price_per_output_unit = price_per_output_unit
        self.pricing_input_unit_type = pricing_input_unit_type
        self.pricing_output_unit_type = pricing_output_unit_type
        self.status = status

    @classmethod
    def from_json(cls, data: dict):
        """
        Create a Model instance from a JSON dictionary.
        """
        return cls(
            provider=data.get("provider", "Unknown"),
            category=data.get("category", "Unknown"),
            model_name=data.get("model_name", "Unknown"),
            price_per_input_unit=data.get("price_per_input_unit", 0.0),
            price_per_output_unit=data.get("price_per_output_unit", 0.0),
            pricing_input_unit_type=data.get("pricing_input_unit_type", "Unknown"),
            pricing_output_unit_type=data.get("pricing_output_unit_type", "Unknown"),
            status=data.get("status", "Unknown"),
        )

    def __repr__(self):
        return (
            f"Model(provider={self.provider}, category={self.category}, model_name={self.model_name}, "
            f"price_per_input_unit={self.price_per_input_unit}, price_per_output_unit={self.price_per_output_unit}, "
            f"pricing_input_unit_type={self.pricing_input_unit_type}, pricing_output_unit_type={self.pricing_output_unit_type}, "
            f"status={self.status})"
        )
