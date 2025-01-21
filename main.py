import streamlit as st
from PIL import Image, ImageOps
import io
import base64
from pillow_heif import register_heif_opener
from together import Together
from openai import OpenAI
import re

import pandas as pd

# Register HEIF opener to support .HEIC images
register_heif_opener()

# Streamlit app title
st.title("Vision Image Uploader")

# Initialize the session state for the uploader key


def calculate_score(student_answers, correct_answers):
    score = 0
    incorrect_questions = {}
    skipped_questions = set(correct_answers.keys()) - set(student_answers.keys())

    for question, correct_answer in correct_answers.items():
        if question in student_answers:
            if student_answers[question].upper() == correct_answer.upper():
                score += 1
            else:
                incorrect_questions[question] = {
                    "Student Answer": student_answers[question],
                    "Correct Answer": correct_answer
                }
    return score, incorrect_questions, skipped_questions
# Allow image upload
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

def reset_uploader():
    """Function to reset the file uploader."""
    st.session_state["uploader_key"] += 1

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "heic"], key=f"uploader_{st.session_state['uploader_key']}",)

if uploaded_file is not None:
    # Add a button to clear/reset the uploader
    if st.button("Upload ảnh khác"):
        reset_uploader()
        st.rerun()
    try:
        # Open the image
        image = Image.open(uploaded_file)

        # Handle image orientation (fix rotation issues)
        image = ImageOps.exif_transpose(image)

        # Resize the image to a manageable size (optional)
        max_size = 2048  # Max width or height
        image.thumbnail((max_size, max_size))

        # Display the uploaded image
        st.image(image, caption='Uploaded Image.', use_container_width=True)

        # Convert the image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")  # Save as JPEG
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Initialize Together client
        

        # Define the query
        # query = "What is in this image?"  # Replace with your desired query
        query = """
        Instructions:
        
        1. Extract the student's multiple-choice answers from the image and provide them as a valid JSON object. The answers may appear in different vietnamese formats (e.g., Câu 1: A, 1. A, Câu 1 - A, Bài 1: A, etc.).
        2. Treat all formats like Bài 1: A, Câu 1: A, or 1. A as referring to Câu X: A for consistency.
        3. Ignore any background noise, blurred lines, or artifacts, and focus only on clearly legible text.
        4. Ensure alignment between question numbers and their corresponding answers.
        5. If the line is misaligned or unreadable, skip it entirely.
        
        Rules:

        1. Output only the JSON object, with no additional text, explanations, or formatting.
        2. Do not include backticks, code blocks, or language specifiers.
        
        Example output:
        {
            "Câu 1": "A",
            "Câu 2": "B",
            "Câu 3": "C",
            "Câu 4": "D"
        }
        Strictly return the JSON object, and nothing else (e.g opening something before json object)
        """


        # Send the image and query to the Together API
        client = Together(api_key=st.secrets["TOGETHER_API_KEY"])
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",  # Replace with your desired model
            messages=[
                # { "role": "system", "content": "You are an assistant that extracts answers from text. Your role is to identify answers written in various formats (e.g., 'Câu 1: A', '1. A', 'Câu 1 - A', 'Bài 1: A') and provide them in a standardized JSON format. Ensure to output only valid answers and skip any incomplete or irrelevant text. Return the results as a JSON object where each key is the question number (e.g., 'Câu 1') and the value is the answer choice (e.g., 'A'). Do not include any additional explanations or formatting." },
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
            max_tokens=2048,
            temperature=0.3,  # Lower temperature for deterministic output
        )

        # client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        # response = client.chat.completions.create(
        #     model="gpt-4o-2024-11-20",
        #     messages=[
        #         # { "role": "system", "content": "You are an assistant that extracts  answers from text. Your role is to identify answers written in various formats (e.g., 'Câu 1: A', '1. A', 'Câu 1 - A', 'Bài 1: A') and provide them in a standardized JSON format. Ensure to output only valid answers and skip any incomplete or irrelevant text. Return the results as a JSON object where each key is the question number (e.g., 'Câu 1') and the value is the answer choice (e.g., 'A'). Do not include any additional explanations or formatting." },
        #         {
        #             "role": "user",
        #             "content": [
        #                 {
        #                     "type": "text",
        #                     "text": query,
        #                 },
        #                 {
        #                     "type": "image_url",
        #                     "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
        #                 },
        #             ],
        #         }
        #     ],
            
        # )

        # Load student answers from the response
        # Display the API response
        st.success("Hình ảnh đã được xử lý")
        st.write("Kết quả lấy từ ảnh:")
        st.write(response.choices[0].message.content) 

        import json
        response_content = response.choices[0].message.content
        print(response_content)
        try:
            student_answers = json.loads(response_content.strip())
        except json.JSONDecodeError:
            st.error("Invalid response format. Please check the input.")
        
        

        # Predefined correct answers
        correct_answers = {
            "Câu 1": "D",
            "Câu 2": "C",
            "Câu 3": "D",
            "Câu 4": "C",
            "Câu 5": "C",
            "Câu 6": "C",
            "Câu 7": "A",
            "Câu 8": "A",
            "Câu 9": "B",
            "Câu 10": "B",
            "Câu 11": "C",
            "Câu 12": "D",
            "Câu 13": "B",
            "Câu 14": "B",
            "Câu 15": "B",
            "Câu 16": "A",
            "Câu 17": "B",
            "Câu 18": "A",
            "Câu 19": "C",
            "Câu 20": "C",
            "Câu 21": "B",
            "Câu 22": "A",
            "Câu 23": "B",
            "Câu 24": "D",
            "Câu 25": "B"
        }

      

        # Calculate results
        score, incorrect_questions, skipped_questions = calculate_score(student_answers, correct_answers)

        # Display results
        st.success("Exam analysis complete!")
        st.write(f":red[Student Score: {score}/{len(correct_answers)}]")

      
        # Display summary table
        results = [
            {
                "Question": question,
                "Student Answer": student_answers.get(question, "Skipped"),
                "Correct Answer": correct_answer,
                "Result": "Correct" if student_answers.get(question, "").upper() == correct_answer.upper() else "Incorrect"
            }
            for question, correct_answer in correct_answers.items()
        ]

        results_df = pd.DataFrame(results)
        # results_df.set_index(results_df.columns[0], inplace=True)
        results_df["Result"] = results_df["Result"].apply(lambda x: "✅ Đúng" if x == "Correct" else "❌ Sai")

        st.write("Summary Table:")
        # st.write(results_df)

        st.dataframe(results_df, use_container_width=True)
        ### chatgpt end here

    except Exception as e:
        st.error(f"An error occurred: {e}")