def generate_markdown_for_models(model) -> str:
    import streamlit as st

    markdown = []
    markdown.append(f"### Model: {model.model_name}")
    markdown.append(f"- **Provider**: {model.provider}")
    markdown.append(f"- **Category**: {model.category}")
    if model.quality:
        markdown.append(f"- **Quality**: {model.quality}")
    if model.size:
        markdown.append(f"- **Size**: {model.size}")

    markdown.append("\n#### Pricing Information")
    markdown.append("| Unit Type | Price Per Input Unit | Price Per Output Unit |")
    markdown.append("|-----------|-----------------------|------------------------|")
    for pricing in model.ai_models_pricing:
        markdown.append(
            f"| {pricing.unit_type} | ${pricing.price_per_input_unit:.6f} | ${pricing.price_per_output_unit:.6f} |"
        )
    with st.expander(f"See {model.model_name} info"):
        st.markdown(f"{"\n".join(markdown)}")
