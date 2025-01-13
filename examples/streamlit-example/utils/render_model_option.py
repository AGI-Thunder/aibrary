from aibrary import AiBrary


def render_model_option(
    aibrary: "AiBrary", category_name: str, selectbox_title="Select a llm model"
):
    from collections import defaultdict

    import streamlit as st

    models = {
        f"{item.model_name}"
        + (f"-{item.size}" if item.size is not None else "")
        + (f",{item.quality}" if item.quality is not None else ""): item
        for item in aibrary.get_all_models(filter_category=category_name)
    }
    if category_name == "chat":
        models.update(
            {
                f"{item.model_name}"
                + (f"-{item.size}" if item.size is not None else "")
                + (f",{item.quality}" if item.quality is not None else ""): item
                for item in aibrary.get_all_models(filter_category="multimodal")
            }
        )
    grouped = defaultdict(list)

    # Group data by the specified field
    for name, obj in models.items():
        grouped[obj.provider].append(name)

    def create_options_with_separator(grouped_dict, separator="🤖"):
        options = []
        for group, items in grouped_dict.items():
            # Add separator for the group with its name
            options.append(f"{separator}{group}")
            # Add items from the group
            options.extend(items)
        if options and options[-1] == separator:
            options.pop()  # Remove the last separator if it exists
        return options

        # Create options with separator

    options = create_options_with_separator(grouped)

    # Display the dropdown
    model_name = st.selectbox(selectbox_title, options, index=1)

    # Handle selection
    if model_name.startswith("🤖"):
        st.warning("This is just a provider, please select a model.")
    return models, model_name
