from dotenv import load_dotenv
load_dotenv()

from query import answer_question

result = answer_question(
    "What do I get with a free Azure account?",
    video_blob_name="c89a56bb-c26e-446d-9a64-e00ef9ecd526.mp4",
)

print("ANSWER:", result["text"])
print("TIMESTAMP:", result["timestamp"])
print("IMAGE PATH:", result["image_path"])
