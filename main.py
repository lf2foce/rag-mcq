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
st.title("Llama Vision Image Uploader")

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
        api_key = st.secrets["TOGETHER_API_KEY"]  # Replace with your actual API key
        client = Together(api_key=api_key)

        # Define the query
        # query = "What is in this image?"  # Replace with your desired query
        query = """
        Extract the student's multiple-choice answers from the image and provide them as a valid JSON object. The answers may appear in different formats (e.g., Câu 1: A, 1. A, Câu 1 - A, etc.).

        Rules:

        Output only the JSON object, with no additional text, explanations, or formatting.
        Do not include backticks, code blocks, or language specifiers.
        Example output:
        {
            "Câu 1": "A",
            "Câu 2": "B",
            "Câu 3": "C",
            "Câu 4": "E"
        }
        Strictly return the JSON object, and nothing else."
        """


        # Send the image and query to the Together API
        # response = client.chat.completions.create(
        #     model="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",  # Replace with your desired model
        #     messages=[
        #         {
        #             "role": "user",
        #             "content": [
        #                 {"type": "text", "text": query},  # Query
        #                 {
        #                     "type": "image_url",
        #                     "image_url": {
        #                         "url": f"data:image/jpeg;base64,{img_base64}"  # Base64-encoded image
        #                     }
        #                 }
        #             ]
        #         }
        #     ],
        #     max_tokens=500
        # )

        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": query,
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                        },
                    ],
                }
            ],
        )

        # student_answers = eval(response.choices[0].message.content)
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

        # Calculate the score
        # score = 0
        # total_questions = len(correct_answers)

        # for question, correct_answer in correct_answers.items():
        #     if question in student_answers:
        #         if student_answers[question].upper() == correct_answer.upper():
        #             score += 1

        # # Display the results
        # st.success("Exam analysis complete!")
        # st.write("Student Answers:", student_answers)
        # st.write(f"Student Score: {score}/{total_questions}")
        
        ### chatgpt

        # Calculate results
        score, incorrect_questions, skipped_questions = calculate_score(student_answers, correct_answers)

        # Display results
        st.success("Exam analysis complete!")
        st.write(f"Student Score: {score}/{len(correct_answers)}")

        if incorrect_questions:
            st.warning("Incorrect Questions:")
            st.write(incorrect_questions)

        if skipped_questions:
            st.info("Skipped Questions:")
            st.write(", ".join(skipped_questions))

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
        st.write("Summary Table:")
        st.write(results_df)
        ### chatgpt end here

        

    except Exception as e:
        st.error(f"An error occurred: {e}")