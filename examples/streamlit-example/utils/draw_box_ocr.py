import io

from PIL import Image, ImageDraw


def draw_box_ocr(image_bytes, ocr_response):
    # Load the image from bytes
    img = Image.open(io.BytesIO(image_bytes))
    draw = ImageDraw.Draw(img)

    # Get the image dimensions
    img_width, img_height = img.size

    # Draw rectangles around bounding boxes
    for bbox in ocr_response.bounding_boxes:
        # Scale bounding box coordinates if they are normalized
        left = bbox.left * img_width  # Assuming normalized coordinates
        top = bbox.top * img_height
        right = (bbox.left + bbox.width) * img_width
        bottom = (bbox.top + bbox.height) * img_height

        # Ensure valid coordinates
        if left < 0 or top < 0 or right > img_width or bottom > img_height:
            continue

        # Ensure y1 >= y0 and x1 >= x0
        if top > bottom or left > right:
            continue

        # Draw rectangle and add text
        draw.rectangle([left, top, right, bottom], outline="red", width=2)
        draw.text(
            (left, max(0, top - 10)), bbox.text, fill="red"
        )  # Add text above the box

    # Save to memory as bytes
    output_bytes = io.BytesIO()
    img.save(output_bytes, format="JPEG")
    output_bytes.seek(0)
    return output_bytes.getvalue()
