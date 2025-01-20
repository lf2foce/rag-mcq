import streamlit as st
from PIL import Image, ImageOps
import io
import base64
from pillow_heif import register_heif_opener
from together import Together

# Register HEIF opener to support .HEIC images
register_heif_opener()

# Streamlit app title
st.title("Llama Vision Image Uploader")

# Allow image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "heic"])

if uploaded_file is not None:
    try:
        # Open the image
        image = Image.open(uploaded_file)

        # Handle image orientation (fix rotation issues)
        image = ImageOps.exif_transpose(image)

        # Resize the image to a manageable size (optional)
        max_size = 1024  # Max width or height
        image.thumbnail((max_size, max_size))

        # Display the uploaded image
        st.image(image, caption='Uploaded Image.', use_container_width=True)

        # Convert the image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")  # Save as JPEG
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Initialize Together client
        api_key = "acccf8bfdc8ab95ed846c49dd53e14a3da5bb05355a8213950b6da5f51d6b3c6"  # Replace with your actual API key
        client = Together(api_key=api_key)

        # Define the query
        query = "What is in this image?"  # Replace with your desired query

        # Send the image and query to the Together API
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",  # Replace with your desired model
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},  # Query
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"  # Base64-encoded image
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )

        # Display the API response
        st.success("API call successful!")
        st.write("Here is the result:")
        st.write(response.choices[0].message.content)  # Display the response content

    except Exception as e:
        st.error(f"An error occurred: {e}")