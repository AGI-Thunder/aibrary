from collections import defaultdict

from aibrary.resources.models import Model


def get_min_model_by_size(models: list[Model]):
    grouped_models = defaultdict(list)
    for model in models:
        model_name = model.model_name
        grouped_models[model_name].append(
            model
        )  # (model_name , model.quality) key of the dict could be name,quality

    min_models = []
    for model_name, models in grouped_models.items():
        min_model = min(
            models, key=lambda m: int(m.size.split("x")[0])
        )  # Assuming `model` has a `size` attribute
        min_models.append(min_model)

    return min_models
