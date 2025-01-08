import io

from PIL import Image, ImageDraw


def draw_box_object_detection(image_bytes, detection_response):
    # Load the image from bytes
    img = Image.open(io.BytesIO(image_bytes))
    draw = ImageDraw.Draw(img)

    # Get the image dimensions
    img_width, img_height = img.size

    # Draw rectangles around bounding boxes
    for item in detection_response.items:
        # Check for None values and skip invalid items
        if (
            item.x_min is None
            or item.y_min is None
            or item.x_max is None
            or item.y_max is None
        ):
            print(f"Skipping item with None values: {item}")
            continue

        try:
            # Convert coordinates to float and scale if necessary
            x_min = (
                float(item.x_min) * img_width
                if isinstance(item.x_min, (str, float))
                else int(item.x_min)
            )
            y_min = (
                float(item.y_min) * img_height
                if isinstance(item.y_min, (str, float))
                else int(item.y_min)
            )
            x_max = (
                float(item.x_max) * img_width
                if isinstance(item.x_max, (str, float))
                else int(item.x_max)
            )
            y_max = (
                float(item.y_max) * img_height
                if isinstance(item.y_max, (str, float))
                else int(item.y_max)
            )
        except ValueError:
            print(f"Skipping item with invalid coordinates: {item}")
            continue

        # Debugging: Print the bounding box values
        print(
            f"Bounding box: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}, label={item.label}"
        )

        # Ensure valid coordinates
        if x_min < 0 or y_min < 0 or x_max > img_width or y_max > img_height:
            print("Skipping invalid bounding box.")
            continue

        # Ensure x_max >= x_min and y_max >= y_min
        if x_max < x_min or y_max < y_min:
            print("Skipping inverted bounding box.")
            continue

        # Draw rectangle and add text
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
        draw.text(
            (x_min, max(0, y_min - 10)),
            f"{item.label} ({item.confidence:.2f})",
            fill="red",
        )  # Add label and confidence

    # Save to memory as bytes
    output_bytes = io.BytesIO()
    img.save(output_bytes, format="JPEG")
    output_bytes.seek(0)
    return output_bytes.getvalue()
