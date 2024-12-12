from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch
import base64
import io
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware
import os
# Initialize the app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)
# Load the Qwen model
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype="auto",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct"
)

# Configure Google Gemini API (make sure to set your valid API key)
genai.configure(api_key="AIzaSyDnqbvPV1-_VfT0kkn69EnstsA0mdFgSw8")  # Replace with your key

@app.get("/")
async def default_get():
    return JSONResponse(content={"message": "Hello, World!"})

@app.post("/process-image/")
async def process_image(file: UploadFile = File(...)):
    try:
        # Read image data
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # Prepare input prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "number", "text": "Response with number of stories in integer type if image is building."}
                ]
            }
        ]

        text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

        # Process inputs for Qwen model
        inputs = processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt"
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = inputs.to(device)

        # Generate output from Qwen model
        output_ids = model.generate(**inputs, max_new_tokens=1024)
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        # Encode image data for Google Gemini API
        encoded_image = base64.b64encode(image_data).decode('utf-8')

        # Prepare prompt for Google Gemini API
        prompt = (
            f"{output_text[0]} \n"
            "Provided this from my machine learning model, it describes the image as a building. "
            "Enhance the output in JSON format by providing the number of floors, correct it if the model predicts wrongly, "
            "and include the height and width of the building in meters."
        )

        # Prepare payload for Gemini
        payload = {
            "mime_type": "image/jpeg",
            "data": encoded_image
        }

        # Generate response from Gemini
        response = genai.GenerativeModel(model_name="gemini-1.5-pro").generate_content(
            [payload, prompt]
        )
        #print(response.text)
        # Return the Gemini response as a JSON
        # return JSONResponse(content={"response": response.text, "model_output": output_text[0]}
        #return JSONResponse(content=[response.text, output_text[0]])
        import json
        import re

        try:
            # Assume gemini_output_raw is the raw text from the response
            gemini_output_raw = response._result.candidates[0].content.parts[0].text

            # Step 1: Remove markdown-like delimiters
            clean_text = gemini_output_raw.strip("```json").strip("```")

            # Step 2: Remove inline comments (e.g., `// Approximate`)
            clean_text = re.sub(r"//.*", "", clean_text)

            # Step 3: Parse the cleaned JSON
            parsed_output = json.loads(clean_text)

            # Access the parsed data
            description = parsed_output["description"]
            floors = parsed_output["floors"]
            height = parsed_output["height"]["value"]
            width = parsed_output["width"]["value"]

            # # Print parsed data
            # print(f"Description: {description}")
            # print(f"Floors: {floors}")
            # print(f"Height: {height} meters")
            # print(f"Width: {width} meters")
            #
            # print("Parsed JSON Object:")
            #print(json.dumps(parsed_output, indent=4))
            return json.dumps(parsed_output, indent=4)

        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)
        except Exception as e:
            print("Unexpected error:", e)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
