from fastapi import FastAPI
from rembg import remove
from PIL import Image

import gradio as gr
import io
import base64

def rembg_api(_: gr.Blocks, app: FastAPI):
    @app.get("/sdapi/v1/rembg/version")
    async def version():
        return {"version": "1.0"}

    @app.post("/sdapi/v1/rembg/remove_background")
    async def remove_background(image_data: dict):
        try:
            # Read upload image
            base64_data = image_data.get("image")
            if not base64_data:
                raise ValueError("No image data provided")
            
            # Convert base64 to image
            content = base64.b64decode(base64_data)
            input_image = Image.open(io.BytesIO(content))

            # Remove background
            output_image = remove(input_image)

            # Convert to byte and return
            output_byte_array = io.BytesIO()
            output_image.save(output_byte_array, format="PNG")

            # Convert to base64
            output_base64 = base64.b64encode(output_byte_array.getvalue()).decode("utf-8")

            return {"success": True, "image": output_base64}
        except Exception as e:
            return {"success": False, "error": str(e)}

try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(rembg_api)
except:
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)